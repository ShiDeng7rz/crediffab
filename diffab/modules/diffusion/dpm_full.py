import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from diffab.modules.encoders.ga import GAEncoder
from diffab.modules.diffusion.transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3)  # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)  # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)  # (*, )
    return loss


class EpsilonNet(nn.Module):
    """
        EpsilonNet: 在扩散模型中扮演核心去噪器的角色。
        输入：当前时刻的 noisy 状态 + 条件特征 + 时间步 beta。输出：
          - 位置噪声预测 eps_pos
          - 旋转更新 v_next、R_next
          - 序列去噪后分类分布 c_denoised
    """

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt=None):
        super().__init__()
        if encoder_opt is None:
            encoder_opt = {}

        # 当前时刻的序列状态 embedding：20 AA + padding + 特殊 = 共25
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding

        # 将残基特征与当前序列 embedding 融合
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        # 图注意力（或几何注意力）编码器：融合 R, p, residue/pair 特征
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        # 位置噪声预测网络：输入 [res_feat, t_embed] -> 预测局部坐标增量
        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        # 旋转噪声预测网络：预测四元数增量的三个分量（1ijk
        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        # 序列分类去噪网络：输出 20 类的 softmax 概率
        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20), nn.Softmax(dim=-1)
        )

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res):
        """
        Args:
            v_t:    (N, L, 3) 当前旋转 so3 向量
            p_t:    (N, L, 3) 当前坐标
            s_t:    (N, L)   当前氨基酸索引
            res_feat:   (N, L, D) 条件残基特征
            pair_feat:  (N, L, L, D_pair) 条件对特征
            beta:   (N,)    当前扩散时间步的噪声强度
            mask_generate: (N, L) 待生成／去噪残基掩码
            mask_res:      (N, L) 有效残基掩码

        Returns:
            v_next: (N, L, 3) 更新后的旋转向量
            R_next: (N, L, 3, 3) 更新后的旋转矩阵
            eps_pos: (N, L, 3)  预测的位置噪声
            c_denoised: (N, L, 20) 预测的氨基酸分类分布
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t)  # (N, L, 3, 3)

        # s_t = s_t.clamp(min=0, max=19)  # TODO: clamping is good but ugly.
        # 融合当前残基特征与序列 embedding
        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_t)],
                                                 dim=-1))  # [Important] Incorporate the sequence at the current step.
        # 用 GAEncoder，将几何（R,p）与残基对特征整合，输出上下文感知的残基特征
        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        # 构造时间步 embedding： [beta, sin(beta), cos(beta)]
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)

        # 拼接时序信息
        in_feat = torch.cat([res_feat, t_embed], dim=-1)

        # Position changes
        # ---- 位置噪声预测 ----
        eps_crd = self.eps_crd_net(in_feat)  # (N, L, 3)
        # 将局部噪声向量旋转到全局坐标
        eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
        # 只保留待生成位置的噪声
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        # New orientation
        # ---- 旋转更新 ----
        eps_rot = self.eps_rot_net(in_feat)  # (N, L, 3)
        # 将三个分量视作四元数 ijk 部分，构建旋转矩阵增量 U
        U = quaternion_1ijk_to_rotation_matrix(eps_rot)  # (N, L, 3, 3)
        R_next = R @ U
        # 非生成位置保持原旋转
        v_next = rotation_to_so3vec(R_next)  # (N, L, 3)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        # New sequence categorical distributions
        # ---- 序列分类去噪 ----
        c_denoised = self.eps_seq_net(in_feat)  # Already softmax-ed, (N, L, 20)

        return v_next, R_next, eps_pos, c_denoised


class FullDPM(nn.Module):

    def __init__(
            self,
            res_feat_dim,  # 残基特征维度， 用于EpsilonNet 输入
            pair_feat_dim,  # 残基对特征维度， 用于EpsilonNet 输入
            num_steps,  # 扩散步骤总数T
            eps_net_opt=None,  # EpsilonNet的配置参数
            trans_rot_opt=None,  # 旋转扩散（Transition）超参
            trans_pos_opt=None,  # 位置扩散超参
            trans_seq_opt=None,  # 序列扩散超参
            position_mean=None,  # 位置均值，用于位置归一化
            position_scale=None,  # 位置缩放，用于位置归一化
    ):
        super().__init__()
        if trans_rot_opt is None:
            trans_rot_opt = {}
        if position_scale is None:
            position_scale = [10.0]
        if position_mean is None:
            position_mean = [0.0, 0.0, 0.0]
        if trans_seq_opt is None:
            trans_seq_opt = {}
        if trans_pos_opt is None:
            trans_pos_opt = {}
        if eps_net_opt is None:
            eps_net_opt = {}

        # 构建去噪网络：接受 noisy 状态 + 条件特征，预测各分量的噪声所在
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = num_steps
        # 旋转分量的噪声/去噪 Transition
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        # 位置分量的噪声/去噪 Transition
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        # 序列分量（类别离散）的噪声/去噪 Transition
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        """对位置坐标做简单的 (p - mean) / scale 归一化"""
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        """将归一化后的位置还原回真实坐标尺度"""
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self,
                v_0,  # (N, L, 3) 初始旋转向量 so3vec
                p_0,  # (N, L, 3) 初始坐标
                s_0,  # (N, L) 初始氨基酸类型索引
                res_feat,  # (N, L, D_res) 残基条件特征
                pair_feat,  # (N, L, L, D_pair) 残基对条件特征
                mask_generate,  # (N, L) 标记哪些残基需要生成／去噪
                mask_res,  # (N, L) 有效残基掩码
                denoise_structure,  # bool, 是否去噪结构分量
                denoise_sequence,  # bool, 是否去噪序列分量
                t=None  # 可选：指定的扩散时刻 t
                ):
        N, L = res_feat.shape[:2]
        # 如果未指定 t，则对每个样本随机采一个步骤
        if t is None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        # 对位置归一化
        p_0 = self._normalize_position(p_0)

        # ===== 结构分量噪声注入 =====
        if denoise_structure:
            # Add noise to rotation
            # 将 so3 向量转回旋转矩阵 R_0，用于损失计算
            R_0 = so3vec_to_rotation(v_0)
            # 在 so3 空间中加噪
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            # Add noise to positions
            # 在位置空间中加噪，并返回真实噪声 eps_p
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            # 如果不去噪结构，则保留原始状态
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        # ===== 序列分量噪声注入 =====
        if denoise_sequence:
            # Add noise to sequence
            # 只需获得 noisy 序列状态
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        # beta 用于 Condition in EpsilonNet
        beta = self.trans_pos.var_sched.betas[t]
        # 预测网络：基于 noisy 状态 + 条件，预测各项去噪量和分类后验
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res
        )  # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)

        loss_dict = {}

        # Rotation loss
        # --- 旋转损失：使用余弦距离衡量 R_pred 与 R_0 之间差异 ---
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0)  # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        # Position loss
        # --- 位置损失：MSE between predicted eps_p and true eps_p ---
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        # Sequence categorical loss
        # --- 序列分类损失：KL Divergence between posterior and predicted posterior ---
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred,
            target=post_true,
            reduction='none',
            log_target=False
        ).sum(dim=-1)  # (N, L)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        return loss_dict

    @torch.no_grad()
    def sample(
            self,
            v, p, s,
            res_feat, pair_feat,
            mask_generate, mask_res,
            sample_structure=True, sample_sequence=True,
            pbar=False,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)

            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )  # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t - 1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])  # Move previous states to cpu memory.

        return traj

    @torch.no_grad()
    def optimize(
            self,
            v, p, s,
            opt_step: int,
            res_feat, pair_feat,
            mask_generate, mask_res,
            sample_structure=True, sample_sequence=True,
            pbar=False,
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)

            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )  # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t - 1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])  # Move previous states to cpu memory.

        return traj
