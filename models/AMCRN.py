import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pooling import AttentiveStatsPool, TemporalAveragePooling
from models.pooling import SelfAttentivePooling, TemporalStatisticsPooling

class AMCRN(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(AMCRN, self).__init__()
        self.dilation = dilation
        self.out_channels = out_channels

        # Initial conv before splitting
        self.pre_dilated_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.pre_bn = nn.BatchNorm1d(out_channels)
        self.pre_relu = nn.ReLU()

        # Shared conv for each subfeature
        self.sub_conv = nn.Conv1d(out_channels // 8, out_channels // 8, kernel_size=3, padding=dilation, dilation=dilation)

        # Extra conv after concatenation (NEW)
        self.post_concat_dilated_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)

        self.final_bn = nn.BatchNorm1d(out_channels)

        # Temporal attention module
        self.attn_conv = nn.Conv1d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def temporal_attention(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, T]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, T]
        attn_input = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, T]
        attn_coeff = self.sigmoid(self.attn_conv(attn_input))  # [B, 1, T]
        return x * attn_coeff  # Broadcast across channels

    def forward(self, x):
        residual = x  # Save original input

        # Step 1: Initial processing
        x = self.pre_dilated_conv(x)
        x = self.pre_bn(x)
        x = self.pre_relu(x)

        # Step 2: Split and process subfeatures
        subfeatures = torch.chunk(x, chunks=8, dim=1)
        processed = []

        for i in range(8):
            if i == 0:
                out = self.sub_conv(subfeatures[i])
            else:
                combined = subfeatures[i] + processed[i - 1]
                out = self.sub_conv(combined)
            processed.append(out)

        # Step 3: Concatenate and process
        y = torch.cat(processed, dim=1)  # [B, C, T]
        y = self.post_concat_dilated_conv(y)  # NEW conv here
        y = self.final_bn(y)

        # Step 4: Temporal attention
        y = self.temporal_attention(y)

        # Step 5: Residual + ReLU
        y = y + residual
        y = self.relu(y)

        return y


class MKRC(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=256, pooling_type="ASP"):
        super(MKRC, self).__init__()
        self.embd_dim = embd_dim
        self.mk_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=5, stride=1, padding=2) # input: (B, 128, T)
        self.mk_layer2 = torch.nn.Conv1d(512, 512, kernel_size=5, dilation=2, padding=4) #MKRC 2
        self.rnn1 = torch.nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True, dropout=0.2)
        self.rnn2 = torch.nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(512, 512)  # Fully connected layer after BLSTM
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

        self.dilated_conv_block1 = AMCRN(in_channels=512, out_channels=512, dilation=2)
        self.dilated_conv_block2 = AMCRN(in_channels=512, out_channels=512, dilation=3)
        self.dilated_conv_block3 = AMCRN(in_channels=512, out_channels=512, dilation=4)

        if pooling_type == "ASP":
            self.pooling = AttentiveStatsPool(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear1 = nn.Linear(channels * 2, embd_dim)
            self.bn7 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

    """def forward(self, x):
        
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        
        x = x.squeeze(1).to("cuda")
        x = x.transpose(1, 2)
        x = self.mk_layer1(x)
        #x = self.mk_layer2(x) #MKRC 2
        x = x.transpose(1, 2)  # From [B, 512, T] → [B, T, 512] #MKRC 3
        x, _ = self.rnn(x) #MKRC 3
        x = x.transpose(1, 2)  # From [B, T, 512] → [B, 512, T] #MKRC 3
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dilated_conv_block1(x)
        x = self.dilated_conv_block2(x)
        x = self.dilated_conv_block3(x)
        x = self.bn5(self.pooling(x))
        out = self.bn7(self.linear1(x))
        return out"""

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.squeeze(1).to("cuda")
        x = x.transpose(1, 2)
        x = self.mk_layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dilated_conv_block1(x)
        x = self.dilated_conv_block2(x)
        x = self.dilated_conv_block3(x)
        x = x.transpose(1, 2)  # From [B, 512, T] → [B, T, 512]
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.fc(x)  # Fully connected layer
        x = x.transpose(1, 2)  # From [B, T, 512] → [B, 512, T]
        x = self.bn5(self.pooling(x))
        out = self.bn7(self.linear1(x))
        return out

def MainModel(**kwargs):

    model =MKRC(
        input_size=112, channels=512, embd_dim=512, pooling_type="ASP"
                )
    return model