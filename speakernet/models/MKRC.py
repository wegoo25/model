import torch
import torch.nn as nn
import torch.nn.functional as F

from speakernet.models.pooling import AttentiveStatsPool, TemporalAveragePooling
from speakernet.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DilatedConvBlock, self).__init__()
        self.dilation = dilation
        self.conv = torch.nn.Conv1d(in_channels // 8, out_channels // 8, kernel_size=3, padding=dilation, dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        subfeatures = torch.chunk(x, chunks=8, dim=1)  # Divide into 8 subfeatures
        processed_subfeatures = [] # list to store Processed subfeatures
        for i in range(8):
            combined = subfeatures[i].clone()  # in-place sums below must not alter x (autograd error + corrupts the residual)
            for j in range(1, min(4, i + 1)):  # Combine with up to 3 previous subfeatures
                combined += processed_subfeatures[i - j]
            out = self.conv(combined)
            processed_subfeatures.append(out)
        y = torch.cat(processed_subfeatures, dim=1)  # Concatenate subfeatures

        y = self.batch_norm(y)
        #y = self.relu(y)
        y = y + x  # Add the input feature (S)
        #y = self.batch_norm(y)
        y = self.relu(y)
        return y


class MKRC(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=256, pooling_type="ASP"):
        super(MKRC, self).__init__()
        self.embd_dim = embd_dim
        self.mk_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=5, stride=1, padding=2) # input: (B, 128, T)
        self.mk_layer2 = torch.nn.Conv1d(512, 512, kernel_size=5, dilation=2, padding=4) #MKRC 2
        # self.rnn = torch.nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True) #MKRC 3
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

        self.dilated_conv_block1 = DilatedConvBlock(in_channels=512, out_channels=512)
        self.dilated_conv_block2 = DilatedConvBlock(in_channels=512, out_channels=512)
        self.dilated_conv_block3 = DilatedConvBlock(in_channels=512, out_channels=512)
        self.dilated_conv_block4 = DilatedConvBlock(in_channels=512, out_channels=512)
        self.dilated_conv_block5 = DilatedConvBlock(in_channels=512, out_channels=512)
        self.dilated_conv_block6 = DilatedConvBlock(in_channels=512, out_channels=512)
        self.dilated_conv_block7 = DilatedConvBlock(in_channels=512, out_channels=512)
        self.dilated_conv_block8 = DilatedConvBlock(in_channels=512, out_channels=512)
        # self.dilated_conv_block9 = DilatedConvBlock(in_channels=512, out_channels=512)
        # self.dilated_conv_block10 = DilatedConvBlock(in_channels=512, out_channels=512)

        if pooling_type == "ASP":
            self.pooling = AttentiveStatsPool(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear1 = nn.Linear(channels * 2, 256)
            self.bn6 = nn.BatchNorm1d(256)
            self.linear2 = nn.Linear(256 ,embd_dim)
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

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.squeeze(1)
        x = x.transpose(1, 2)
        x = self.mk_layer1(x)
        x = self.mk_layer2(x) #MKRC 2
        # x = x.transpose(1, 2)  # From [B, 512, T] → [B, T, 512] #MKRC 3
        # x, _ = self.rnn(x) #MKRC 3
        # x = x.transpose(1, 2)  # From [B, T, 512] → [B, 512, T] #MKRC 3
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dilated_conv_block1(x)
        x = self.dilated_conv_block2(x)
        x = self.dilated_conv_block3(x)
        x = self.dilated_conv_block4(x)
        x = self.dilated_conv_block5(x)
        x = self.dilated_conv_block6(x)
        x = self.dilated_conv_block7(x)
        x = self.dilated_conv_block8(x)
        # x = self.dilated_conv_block9(x)
        # x = self.dilated_conv_block10(x)
        x = self.bn5(self.pooling(x))
        x = self.bn6(self.linear1(x))
        out = self.bn7(self.linear2(x))
        return out

def MainModel(**kwargs):

    model =MKRC(
        input_size=112, channels=512, embd_dim=512, pooling_type="ASP"
                )
    return model