import torch
import torch.nn as nn 
import torch.nn.functional as F

class EncCNN1d(nn.Module): #对语音特征做卷积的网络
    def __init__(self, input_dim=130, channel=128, dropout=0.3):
        super(EncCNN1d, self).__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(input_dim, channel, 10, stride=2, padding=4), #in_channels, out_channels, kernel_size, stride, padding #600->300
            nn.BatchNorm1d(channel),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(channel, channel*2, 5, stride=2, padding=2), #300->150
            nn.BatchNorm1d(channel*2),
            nn.LeakyReLU(0.3, inplace=True),
            #nn.Conv1d(channel*2, channel*4, 5, stride=2, padding=2), #150->75
            #nn.BatchNorm1d(channel*4),
            #nn.LeakyReLU(0.3, inplace=True),

            #nn.Conv1d(channel*4, channel*2, 3, stride=1, padding=1), #75->75

            nn.Conv1d(channel*2, channel*2, 5, stride=2, padding=2), #150->75
            nn.BatchNorm1d(channel*2),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv1d(channel*2, channel, 3, stride=1, padding=1), #75->75
        )
        self.dp = nn.Dropout(dropout)

    def forward(self, wav_data):
        # wav_data of shape [bs, seq_len, input_dim]
        out = self.feat_extractor(wav_data.transpose(1, 2))
        out = out.transpose(1, 2)       # to (batch x seq x dim)
        out = self.dp(out)
        return out  

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3, \
                                pooling=False, pool_size=None, pool_stride=None):
        super().__init__()
        self.conv1 = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.3),
            #nn.Dropout(dropout)
        ]
        self.conv2 = [
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.3),
            #nn.Dropout(dropout)
        ]
        if pooling:
            self.conv1.append(
                nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride, padding=(kernel_size-1)//2)
            )
            self.conv2.append(
                nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride, padding=(kernel_size-1)//2)
            )
        self.conv1 = nn.Sequential(*self.conv1)
        self.conv2 = nn.Sequential(*self.conv2)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3, downsample=True, \
                                pooling=True, pool_size=None, pool_stride=None):
        super().__init__()
        self.conv_down = nn.Conv1d(in_channels, out_channels, kernel_size, stride=2 if downsample else 1)
        self.block1 = ResBlock(out_channels, out_channels, kernel_size, dropout)
        self.block2 = ResBlock(out_channels, out_channels, kernel_size, dropout)
    
    def forward(self, x):
        downsample = self.conv_down(x)
        out = self.block1(downsample)
        out = self.block2(downsample)
        return out
        
class ResNetEncoder(nn.Module):
    def __init__(self, input_dim=130, channels=128):
        super().__init__()
        self.cnn_in = nn.Conv1d(input_dim, channels, kernel_size=10, stride=2)
        self.cnn_block1 = CNNBlock(channels, 2*channels, kernel_size=5, downsample=True)
        self.cnn_block2 = CNNBlock(2*channels, 4*channels, kernel_size=5, downsample=True)
        self.cnn_block3 = CNNBlock(4*channels, 4*channels, kernel_size=3, downsample=False)
        self.cnn_block4 = CNNBlock(4*channels, 2*channels, kernel_size=3, downsample=False)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        _in = self.cnn_in(x)
        out = self.cnn_block1(_in)
        out = self.cnn_block2(out)
        out = self.cnn_block3(out)
        out = self.cnn_block4(out)
        return out.transpose(1, 2)

if __name__ == '__main__':
    # x = torch.rand(2, 600, 130)
    # net = EncCNN1d()
    # out = net(x)
    # print(net)
    # print(out.size())
    # num_params = 0
    # for param in net.parameters():
    #     num_params += param.numel()
    # print('Total number of parameters : %.3f M' % (num_params / 1e6))

    x = torch.rand(2, 130, 600)
    input_dim = 130
    channel = 128
    net1 = nn.Conv1d(input_dim, channel, 10, stride=2, padding=4)
    net2 = nn.Conv1d(channel, channel*2, 5, stride=2, padding=2)
    #net3 = nn.Conv1d(channel*2, channel*4, 5, stride=2, padding=2)
    net3 = nn.Conv1d(channel*2, channel*2, 5, stride=2, padding=2)
    #net4 = nn.Conv1d(channel*4, channel*2, 3, stride=1, padding=1)
    net4 = nn.Conv1d(channel*2, channel, 3, stride=1, padding=1)
    out1 = net1(x)
    out2 = net2(out1)
    out3 = net3(out2)
    out4 = net4(out3)
    print(x.shape)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)

    # x = torch.rand(2, 600, 130)
    # net = ResNetEncoder(channels=64)
    # out = net(x)
    # print(net)
    # print(out.size())
    # num_params = 0
    # for param in net.parameters():
    #     num_params += param.numel()
    # print('Total number of parameters : %.3f M' % (num_params / 1e6))