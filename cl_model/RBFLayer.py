import torch
import torch.nn as nn


class RBFLayer(nn.Module):
    """
    Radial Basis Function (RBF) encoding for distances.
    """

    def __init__(self,
                 num_centers: int = 16,
                 min_dist: float = 0.0,
                 max_dist: float = 20.0,
                 width: float = None):
        """
        Args:
            num_centers: number of Gaussian kernels (output dimension).
            min_dist: lower bound of distance centers.
            max_dist: upper bound of distance centers.
            width: Gaussian σ; if None, set to (max_dist - min_dist) / num_centers.
        """
        super().__init__()
        centers = torch.linspace(min_dist, max_dist, num_centers)
        self.register_buffer('centers', centers)  # fixed, not learnable
        self.width = (max_dist - min_dist) / num_centers if width is None else width

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: Tensor of shape [...], distances.

        Returns:
            Tensor of shape [..., num_centers], where
            output[..., k] = exp( - (dist - centers[k])^2 / width^2 )
        """
        # diff: [..., num_centers]
        diff = dist.unsqueeze(-1) - self.centers
        return torch.exp(-(diff ** 2) / (self.width ** 2))


def batched_pooled_rbf(coord: torch.Tensor,
                       mask: torch.Tensor,
                       edge_index: torch.LongTensor,
                       rbf_layer: RBFLayer) -> torch.Tensor:
    """
    Args:
        coord:      [N_res, 14, 3]
        mask:       [N_res, 14]         (0/1)
        edge_index: [2, E]             pairs of residue indices
        rbf_layer:  RBFLayer instance

    Returns:
        edge_feat: [E, num_centers]
    """
    row, col = edge_index  # each [E]

    # gather per-edge coords & masks
    ci = coord[row]  # [E, 14, 3]
    cj = coord[col]  # [E, 14, 3]
    mi = mask[row].float()  # [E, 14]
    mj = mask[col].float()  # [E, 14]

    # 1) pairwise distances for each edge: [E,14,14]
    #    ci.unsqueeze(2): [E,14,1,3], cj.unsqueeze(1): [E,1,14,3]
    d_ij = torch.norm(ci.unsqueeze(2) - cj.unsqueeze(1), dim=-1)

    # 2) valid atom‐atom mask: [E,14,14]
    valid = (mi.unsqueeze(2) * mj.unsqueeze(1)).bool()

    # 3) RBF encode: [E,14,14,num_centers]
    rbf_ij = rbf_layer(d_ij)

    # 4) mask out invalid pairs
    rbf_ij = rbf_ij * valid.unsqueeze(-1)

    # 5) sum & average over the two atom dimensions
    sum_rbf = rbf_ij.sum(dim=(1, 2))  # [E, num_centers]
    count = valid.sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1).float()  # [E,1]

    edge_feat = sum_rbf / count  # [E, num_centers]
    return edge_feat


# ===== Example =====
if __name__ == "__main__":
    N_res = 5
    coord = torch.rand(N_res, 14, 3) * 20
    mask = (torch.rand(N_res, 14) > 0.2).long()

    # make a small edge_index: fully connected minus self
    rows, cols = [], []
    for i in range(N_res):
        for j in range(N_res):
            if i != j:
                rows.append(i)
                cols.append(j)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)

    # instantiate
    rbf_layer = RBFLayer(num_centers=16, min_dist=0.0, max_dist=20.0)

    # compute per-edge RBF features
    edge_features = batched_pooled_rbf(coord, mask, edge_index, rbf_layer)
    print("edge_features.shape =", edge_features.shape)  # should be [N_res*(N_res-1), 16]
