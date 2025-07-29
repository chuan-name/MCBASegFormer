import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight * x



class GlobalContextAwareModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalContextAwareModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)  # rate=1
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)  # rate=3
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5)  # rate=5

        self.conv_1x1 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)


        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)


        self.fc1 = nn.Linear(out_channels*3, out_channels // 16)
        self.fc2 = nn.Linear(out_channels // 16, out_channels*3)

        self.cross_attn = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, 2, 1),
            nn.Sigmoid()
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(2*out_channels, out_channels, 1)

    def forward(self, x):
        D1 = self.conv1(x)
        D2 = self.conv2(x)
        D3 = self.conv3(x)


        features = torch.cat([D1, D2, D3], dim=1)
        branch1 = self.conv_1x1(features)
        branch1 = self.softmax(branch1)

        weighted_D1 = branch1 * D1
        weighted_D2 = branch1 * D2
        weighted_D3 = branch1 * D3
        S1 = weighted_D1 + weighted_D2 + weighted_D3
        gap = self.global_avg_pool(features)

        gap = gap.view(gap.size(0), -1)

        gap = self.fc1(gap)
        gap = F.relu(gap)
        Wc = self.fc2(gap)
        Wc = torch.sigmoid(Wc)


        if Wc.dim() == 2:
            Wc = Wc.view(Wc.size(0), Wc.size(1), 1, 1)
        if Wc.shape[2:] != D1.shape[2:]:
            Wc = F.interpolate(Wc, size=D1.shape[2:], mode='nearest')
        Wc_D1, Wc_D2, Wc_D3 = torch.split(Wc, D1.shape[1], dim=1)
        weighted_D1 = Wc_D1 * D1
        weighted_D2 = Wc_D2 * D2
        weighted_D3 = Wc_D3 * D3
        S2 = weighted_D1 + weighted_D2 + weighted_D3

        cross_weights = self.cross_attn(torch.cat([S1, S2], dim=1))
        w_s, w_c = cross_weights.chunk(2, dim=1)

        enhanced_S1 = S1 * w_s * torch.sigmoid(S2)
        enhanced_S2 = S2 * w_c * torch.sigmoid(S1)
        output = torch.cat([enhanced_S1, enhanced_S2], dim=1)
        x_res = self.res_conv(x)
        output = self.out_conv(output)
        output=output+x_res
        return output

