import torch
import torch.nn as nn

from diffab.modules.common.geometry import construct_3d_basis
# 把旋转矩阵转换为 so3 空间向量
from diffab.modules.common.so3 import rotation_to_so3vec
from diffab.modules.diffusion.dpm_full import FullDPM
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms  # 15
}


@register_model('diffab')
class DiffusionAntibodyDesign(nn.Module):
    """抗体设计的扩散模型：联合结构与序列的条件扩散生成"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 根据分辨率选 atom 数目，用于 ResidueEmbedding 和 PairEmbedding
        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        # 单残基嵌入（将氨基酸、结构信息编码为向量）
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        # 残基对嵌入（编码残基之间的几何和序列关系）
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)

        # 构建完整的扩散模型：输入残基/对特征，输出结构+序列的去噪预测
        self.diffusion = FullDPM(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            **cfg.diffusion,
        )

    def encode(self, batch, remove_structure, remove_sequence):
        """
        对 batch 中的蛋白质结构和序列进行嵌入编码，并构造 3D 基底。

        Args:
            batch: 包含以下 key 的字典
                - 'aa': (N, L) 氨基酸类别索引
                - 'res_nb', 'chain_nb': (N, L) 残基编号、链编号
                - 'pos_heavyatom': (N, L, A, 3) 各重原子坐标
                - 'mask_heavyatom': (N, L, A) 原子有效掩码
                - 'fragment_type': (N, L) 片段类型等
                - 'mask_heavyatom' 和 'generate_flag' 用于构造 context mask
            remove_structure: 是否在编码时隐藏已有结构
            remove_sequence: 是否在编码时隐藏已有序列

        Returns:
            res_feat:  (N, L, res_feat_dim)  单残基特征
            pair_feat: (N, L, L, pair_feat_dim)  残基对特征
            R:         (N, L, 3, 3)             每个残基的局部 3D 基底
            p:         (N, L, 3)                每个残基 CA 原子的坐标
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        # context_mask: 同时满足有 CA 原子且非生成目标的残基被视为上下文
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA],
            ~batch['generate_flag']  # Context means ``not generated''
        )

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        # 残基级别特征嵌入
        res_feat = self.residue_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            fragment_type=batch['fragment_type'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        # 残基对级别特征嵌入
        pair_feat = self.pair_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        # 构造每个残基的局部 3D 坐标系：用 CA, C, N 三点
        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        # p: 每个残基的参考点，这里选 CA 原子
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p

    def forward(self, batch):
        """
           训练时前向调用：
           1. 编码结构/序列上下文
           2. 将基底 R 转为 so3 向量 v_0，s_0 为序列初始状态
           3. 调用扩散模型计算去噪损失
        """
        mask_generate = batch['generate_flag']  # 要生成的位置
        mask_res = batch['mask']  # 有效的残基掩码

        # 编码时屏蔽训练中剔除的结构/序列
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure=self.cfg.get('train_structure', True),
            remove_sequence=self.cfg.get('train_sequence', True)
        )
        # 将局部旋转矩阵 R_0 转为 so(3) 空间向量 v_0
        v_0 = rotation_to_so3vec(R_0)
        # 初始序列状态 s_0 就是氨基酸索引
        s_0 = batch['aa']

        # 调用 FullDPM 扩散模型：同时输出结构和序列的去噪损失
        loss_dict = self.diffusion(
            v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
            denoise_structure=self.cfg.get('train_structure', True),
            denoise_sequence=self.cfg.get('train_sequence', True),
        )
        return loss_dict

    @torch.no_grad()
    def sample(
            self,
            batch,
            sample_opt=None
    ):
        if sample_opt is None:
            sample_opt = {
                'sample_structure': True,
                'sample_sequence': True,
            }
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure=sample_opt.get('sample_structure', True),
            remove_sequence=sample_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        traj = self.diffusion.sample(v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, **sample_opt)
        return traj

    @torch.no_grad()
    def optimize(
            self,
            batch,
            opt_step,
            optimize_opt=None
    ):
        if optimize_opt is None:
            optimize_opt = {
                'sample_structure': True,
                'sample_sequence': True,
            }
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure=optimize_opt.get('sample_structure', True),
            remove_sequence=optimize_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(v_0, p_0, s_0, opt_step, res_feat, pair_feat, mask_generate, mask_res,
                                       **optimize_opt)
        return traj
