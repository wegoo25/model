import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pooling import AttentiveStatsPool

class GraphAttention1D(nn.Module):
    def __init__(self, in_channels, k=2):
        super(GraphAttention1D, self).__init__()
        self.k = k
        self.W = nn.Linear(in_channels, in_channels)
        self.a = nn.Linear(in_channels * 2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.size()
        x = x.transpose(1, 2)  # [B, T, C]
        Wh = self.W(x)         # [B, T, C]
        outputs = []
        for t in range(T):
            neighbors = []
            for dt in range(-self.k, self.k + 1):
                idx = t + dt
                if 0 <= idx < T:
                    neighbors.append(Wh[:, idx, :])
            h_self = Wh[:, t, :].unsqueeze(1).repeat(1, len(neighbors), 1)  # [B, num_neighbors, C]
            h_neighbors = torch.stack(neighbors, dim=1)                     # [B, num_neighbors, C]
            attn_input = torch.cat([h_self, h_neighbors], dim=2)            # [B, num_neighbors, 2C]
            e = self.leakyrelu(self.a(attn_input)).squeeze(-1)              # [B, num_neighbors]
            alpha = F.softmax(e, dim=1)                                     # [B, num_neighbors]
            h_prime = torch.sum(alpha.unsqueeze(-1) * h_neighbors, dim=1)   # [B, C]
            outputs.append(h_prime)
        out = torch.stack(outputs, dim=1)  # [B, T, C]
        return out.transpose(1, 2)         # [B, C, T]

class TemporalSelfAttention(nn.Module):
    def __init__(self, channels):
        super(TemporalSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        out, _ = self.attn(x, x, x)
        return out.transpose(1, 2)

class DynamicCrossScaleFusion(nn.Module):
    def __init__(self, num_scales, channels):
        super(DynamicCrossScaleFusion, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(num_scales * channels, num_scales, 1),
            nn.Softmax(dim=1)
        )
        self.channels = channels
        self.num_scales = num_scales

    def forward(self, x):
        # x: [B, num_scales*channels, T]
        B, C, T = x.size()
        x_reshaped = x.view(B, self.num_scales, self.channels, T)  # [B, num_scales, channels, T]
        attn_scores = self.attn(x)  # [B, num_scales, T]
        attn_scores = attn_scores.unsqueeze(2)  # [B, num_scales, 1, T]
        fused = (x_reshaped * attn_scores).sum(dim=1)  # [B, channels, T]
        return fused

class MKRC_GAT(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=256, pooling_type="ASP", temporal_model='tcn'):
        super(MKRC_GAT, self).__init__()
        self.embd_dim = embd_dim
        self.mk_layer1 = nn.Conv1d(input_size, channels, kernel_size=5, stride=1, padding=2)
        self.mk_layer2 = nn.Conv1d(channels, channels, kernel_size=5, dilation=2, padding=4)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        # Graph Attention block for each sub-feature
        self.gat_blocks = nn.ModuleList([GraphAttention1D(channels // 8) for _ in range(8)])

        # Temporal modeling block
        if temporal_model == 'tcn':
            self.temporal = TemporalConvBlock(channels)
        else:
            self.temporal = TemporalSelfAttention(channels)

        self.dcsf = DynamicCrossScaleFusion(num_scales=8, channels=channels // 8)

        # Pooling and projection (as before)
        if pooling_type == "ASP":
            self.pooling = AttentiveStatsPool(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear1 = nn.Linear(channels * 2, 256)
            self.bn6 = nn.BatchNorm1d(256)
            self.linear2 = nn.Linear(256, embd_dim)
            self.bn7 = nn.BatchNorm1d(embd_dim)
        # ... (other pooling types as in your code)

    def forward(self, x):
        x = x.squeeze(1).to("cuda")
        x = x.transpose(1, 2)
        x = self.mk_layer1(x)
        x = self.mk_layer2(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Split into 8 sub-features
        subfeatures = torch.chunk(x, 8, dim=1)
        processed = []
        for i, sf in enumerate(subfeatures):
            processed.append(self.gat_blocks[i](sf))
        x = torch.cat(processed, dim=1)
        x = self.dcsf(x)
        x = self.temporal(x)

        # Pooling and projection
        x = self.bn5(self.pooling(x))
        x = self.bn6(self.linear1(x))
        out = self.bn7(self.linear2(x))
        return out

# def MainModel(**kwargs):
#     model =MKRC_GAT(
#         input_size=112, channels=512, embd_dim=256, pooling_type="ASP", temporal_model='tsa'
#     )
#     return model

if __name__ == "__main__":
    torch.manual_seed(0)  # For reproducibility

    B, C, T = 32, 512, 300
    k = 2  # Number of neighbors

    # Create random input tensor with uniform distribution [0,1)
    x = torch.rand(B, C, T)
    print(f"Input x shape: {x.shape}")  # [32, 512, 300]

    # Instantiate GraphAttention1D block
    gat = GraphAttention1D(in_channels=C, k=k)

    # Forward pass
    out = gat(x)

    print(f"Output shape: {out.shape}")  # Should be [32, 512, 300]

    # To debug internal variables, you can temporarily add print statements inside GraphAttention1D.forward:
    # For example, print shapes of Wh, neighbors, attention scores, alpha, h_prime for t=0 or any t.