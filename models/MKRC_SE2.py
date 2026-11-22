import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pooling import AttentiveStatsPool, TemporalAveragePooling
from models.pooling import SelfAttentivePooling, TemporalStatisticsPooling

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DilatedConvBlock, self).__init__()
        self.dilation = dilation
        self.conv2d = nn.Conv2d(1, in_channels, kernel_size=(3, 3), padding=(1, 1))  # 2D convolutional layer
        self.conv = torch.nn.Conv1d(in_channels // 8, out_channels // 8, kernel_size=3, padding=dilation, dilation=dilation)
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 512 // 16),
            nn.ReLU(),
            nn.Linear(512 // 16, 512),
            nn.Sigmoid()
        ) #MKRC_SE
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = x.unsqueeze(1)  # Add a channel dimension for 2D convolution, shape: B x 1 x F x T
        #x = self.conv2d(x)  # Apply 2D convolution
        #x = x.squeeze(1)  # Remove the channel dimension after 2D convolution, shape: B x F x T
        subfeatures = torch.chunk(x, chunks=8, dim=1)  # Divide into 8 subfeatures
        processed_subfeatures = [] # list to store Processed subfeatures
        for i in range(8):
            if i == 0:
                out = self.conv(subfeatures[i])
            elif i == 1:
                combined = subfeatures[i] + processed_subfeatures[i-1]
                out = self.conv(combined)
            else:
                combined = subfeatures[i] + processed_subfeatures[i-1] + processed_subfeatures[i-2]
                out = self.conv(combined)

            processed_subfeatures.append(out) # Append the processed subfeature

        y = torch.cat(processed_subfeatures, dim=1)  # Concatenate subfeatures
        y = y.unsqueeze(1)  # Add a channel dimension for 2D convolution, shape: B x 1 x F x T
        y = self.conv2d(y)  # Apply 2D convolution
        y = y.mean(dim=2)
        y = self.batch_norm(y)
        # SE block
        avg_pool = torch.mean(y, dim=2, keepdim=True)
        max_pool, _ = torch.max(y, dim=2, keepdim=True)  # Shape: B x F x 1
        pool = torch.cat([avg_pool, max_pool], dim=1)  # Concatenate along the channel dimension, shape: B x 2F x 1
        scale = self.fc(pool.squeeze(-1))  # Pass through the fully connected layers, shape: B x 512
        scale = scale.unsqueeze(-1)  # Add the time dimension back, shape: B x 512 x 1
        y = y * scale
        y = y + x  # Add the input feature (S)
        y = self.relu(y)
        return y


class MKRC(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=256, pooling_type="ASP"):
        super(MKRC, self).__init__()
        self.embd_dim = embd_dim
        self.mk_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=5, stride=1, padding=2) # input: (B, 128, T)
        self.mk_layer2 = torch.nn.Conv1d(512, 512, kernel_size=5, dilation=2, padding=4) #MKRC 2
        self.conv2d = nn.Conv2d(1, 512, kernel_size=(3, 3), padding=(1, 1))  # 2D convolutional layer
        # self.rnn = torch.nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True) #MKRC 3
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()

        self.dilated_conv_block1 = DilatedConvBlock(in_channels=512, out_channels=512)
        self.dilated_conv_block2 = DilatedConvBlock(in_channels=512, out_channels=512)
        self.dilated_conv_block3 = DilatedConvBlock(in_channels=512, out_channels=512)

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
        #x = x.squeeze(1).to("cuda")
        x = x.transpose(2, 3).to("cuda")
        x = self.conv2d(x)
        #x = self.mk_layer1(x)
        #x = self.mk_layer2(x) #MKRC 2
        # x = x.transpose(1, 2)  # From [B, 512, T] → [B, T, 512] #MKRC 3
        # x, _ = self.rnn(x) #MKRC 3
        # x = x.transpose(1, 2)  # From [B, T, 512] → [B, 512, T] #MKRC 3
        x = self.bn1(x)
        x = self.relu(x)
        x = x.mean(dim=2)  # → [B, 512, T]
        x = self.dilated_conv_block1(x)
        x = self.dilated_conv_block2(x)
        x = self.dilated_conv_block3(x)
        x = self.bn5(self.pooling(x))
        x = self.bn6(self.linear1(x))
        out = self.bn7(self.linear2(x))
        return out

def MainModel(**kwargs):

    model =MKRC(
        input_size=112, channels=512, embd_dim=512, pooling_type="ASP"
                )
    return model