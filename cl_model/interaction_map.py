import torch
import torch.nn.functional as F


def compute_dis_from_atoms(
        ab_atoms: torch.Tensor,  # (B, M, K, 3), Å
        ag_atoms: torch.Tensor,  # (B, N, K, 3), Å
        ab_mask_res: torch.Tensor,  # (B, M)    残基有效 True/False
        ag_mask_res: torch.Tensor,  # (B, N)
        ab_mask_atom: torch.Tensor,  # (B, M, K) 原子有效 True/False
        ag_mask_atom: torch.Tensor,  # (B, N, K)
        threshold: float = 4.5,  # Å
        block_M: int | None = None,  # 可选：Mi 方向分块，缓解显存 (例如 256)
        block_N: int | None = None,  # 可选：Ni 方向分块
):
    """
    返回：list[Tensor]，每个样本一个 (Mi, Ni) 的 0/1 接触图（float32）
    """
    B, M, K, _ = ab_atoms.shape
    _, N, _, _ = ag_atoms.shape
    out: list[torch.Tensor] = []

    for i in range(B):
        # 先按“残基层”筛选有效残基
        m_res = ab_mask_res[i]  # (M,)
        n_res = ag_mask_res[i]  # (N,)
        if m_res.sum() == 0 or n_res.sum() == 0:
            out.append(torch.zeros((0, 0), device=ab_atoms.device, dtype=ab_atoms.dtype))
            continue

        A = ab_atoms[i][m_res]  # (Mi, K, 3)
        G = ag_atoms[i][n_res]  # (Ni, K, 3)
        A_mask = ab_mask_atom[i][m_res]  # (Mi, K)
        G_mask = ag_mask_atom[i][n_res]  # (Ni, K)

        Mi, Ni = A.size(0), G.size(0)
        cm = torch.full((Mi, Ni), 0.0, device=A.device, dtype=A.dtype)  # contact map

        # 分块（避免一次性构造 (Mi*K, Ni*K) 的大矩阵）
        m_step = Mi if block_M is None else block_M
        n_step = Ni if block_N is None else block_N

        for ms in range(0, Mi, m_step):
            me = min(ms + m_step, Mi)
            A_blk = A[ms:me]  # (mb, K, 3)
            Am_blk = A_mask[ms:me]  # (mb, K)

            for ns in range(0, Ni, n_step):
                ne = min(ns + n_step, Ni)
                G_blk = G[ns:ne]  # (nb, K, 3)
                Gm_blk = G_mask[ns:ne]  # (nb, K)

                mb, nb = A_blk.size(0), G_blk.size(0)

                # (mb*K, 3) vs (nb*K, 3) -> (mb*K, nb*K) -> (mb, K, nb, K)
                d = torch.cdist(A_blk.reshape(-1, 3), G_blk.reshape(-1, 3))
                d = d.view(mb, K, nb, K)

                # 原子级无效项置 inf，再取 amin
                valid = Am_blk[:, :, None, None] & Gm_blk[None, None, :, :]  # (mb,K,nb,K)
                d.masked_fill_(~valid, float("inf"))

                d_min = d.amin(dim=(1, 3))  # (mb, nb)
                cm[ms:me, ns:ne] = (d_min < threshold).float()

        out.append(cm)

    return out


class NodeInteractionBilinearLoss(torch.nn.Module):
    """
    自定义 QK 打分（未 softmax 的 logits），用 BCEWithLogits 对齐结构接触图。
    """

    def __init__(self, hidden_size=256, temperature=0.1, pos_weight=None, device="cpu"):
        super().__init__()
        self.Wq = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.Wk = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.temperature = float(temperature)
        self.pos_weight = pos_weight

    def forward(self, ab_aux, ag_aux,
                ab_atom, ag_atom, ab_mask_res, ag_mask_res, ab_mask_atom, ag_mask_atom,
                threshold=4.5, block_M=None, block_N=None):
        device = ab_atom.device
        B = ab_atom.size(0)

        dis = compute_dis_from_atoms(
            ab_atom, ag_atom, ab_mask_res, ag_mask_res, ab_mask_atom, ag_mask_atom,
            threshold=threshold, block_M=block_M, block_N=block_N
        )

        ab_batch = ab_aux["batch"]
        ag_batch = ag_aux["batch"]
        Aall = ab_aux["node_embeddings"]
        Gall = ag_aux["node_embeddings"]
        d = Aall.size(-1)

        total_loss, n_terms = 0.0, 0

        for k in range(B):
            ab_idx = (ab_batch == k)
            ag_idx = (ag_batch == k)
            if ab_idx.sum() == 0 or ag_idx.sum() == 0:
                continue

            A = Aall[ab_idx]  # (m, d)
            G = Gall[ag_idx]  # (n, d)

            q = self.Wq(A)  # (m, d)
            kvec = self.Wk(G)  # (n, d)
            logits = (q @ kvec.t()) / (d ** 0.5)
            logits = logits / self.temperature  # 真正 logits

            y = dis[k].to(device)  # (m, n)
            m = min(logits.size(0), y.size(0))
            n = min(logits.size(1), y.size(1))
            if m == 0 or n == 0:
                continue
            logits, y = logits[:m, :n], y[:m, :n]

            if self.pos_weight is None:
                pos = y.sum()
                neg = y.numel() - pos
                pw = (neg / (pos + 1e-6)).clamp(max=100.0)
                pw = torch.as_tensor(pw.item(), device=device)
            else:
                pw = torch.tensor(float(self.pos_weight), device=device)

            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            total_loss += loss
            n_terms += 1

        if n_terms == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / n_terms
