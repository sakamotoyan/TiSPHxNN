import torch.nn as nn

'''
    Conv2dBLD
    B: BatchNorm
    L: LeakyReLU
    D: Dropout
'''
class Conv2dNAD_E(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leakiness, dropout_probability):
        super(Conv2dNAD_E, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(leakiness)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class Conv2dNAD_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, leakiness, dropout_probability, order):
        super(Conv2dNAD_D, self).__init__()
        self.order = order
        if self.order == 0:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        else:
            self.bn = nn.BatchNorm2d(in_channels)
            self.activation = nn.LeakyReLU(leakiness)
            self.dropout = nn.Dropout(dropout_probability)
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.order == 0:
            x = self.conv(x)
        else:
            x = self.bn(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.conv(x)
        return x

'''
    Conv2dMBLD
    M: Maxpooling
    B: BatchNorm
    L: LeakyReLU
    D: Dropout
'''
class Conv2dNAMD_E(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leakiness, dropout_probability):
        super(Conv2dNAMD_E, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(leakiness)
        self.maxpool = nn.MaxPool2d(stride)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x

class Conv2dNAMD_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, leakiness, dropout_probability, order):
        super(Conv2dNAMD_D, self).__init__()
        self.order = order
        if self.order == 0:
            self.maxunpool = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding)
        else:
            self.bn = nn.BatchNorm2d(in_channels)
            self.activation = nn.LeakyReLU(leakiness)
            self.maxunpool = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.dropout = nn.Dropout(dropout_probability)
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        if self.order == 0:
            x = self.maxunpool(x)
            x = self.conv(x)
        else:
            x = self.bn(x)
            x = self.activation(x)
            x = self.maxunpool(x)
            x = self.dropout(x)
            x = self.conv(x)
        return x

'''
    DoubleConv2dMBLD
    M: Maxpooling
    B: BatchNorm
    L: LeakyReLU
    D: Dropout
'''
class DoubleConv2dNADM_E(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leakiness, dropout_probability):
        super(DoubleConv2dNADM_E, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.LeakyReLU(leakiness)
        self.dropout1 = nn.Dropout(dropout_probability)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU(leakiness)
        self.maxpool = nn.MaxPool2d(stride)
        self.dropout2 = nn.Dropout(dropout_probability)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.maxpool(x)
        x = self.dropout2(x)
        return x
    
class DoubleConv2dNADM_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, leakiness, dropout_probability, order):
        super(DoubleConv2dNADM_D, self).__init__()
        self.order = order
        if self.order == 0:
            self.maxunpool = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.activation2 = nn.LeakyReLU(leakiness)
            self.dropout2 = nn.Dropout(dropout_probability)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, padding=padding)
        else:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.activation1 = nn.LeakyReLU(leakiness)
            self.maxunpool = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.dropout1 = nn.Dropout(dropout_probability)
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.activation2 = nn.LeakyReLU(leakiness)
            self.dropout2 = nn.Dropout(dropout_probability)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, padding=padding)
        
        
    def forward(self, x):
        if self.order == 0:
            x = self.maxunpool(x)
            x = self.conv1(x)
            x = self.bn2(x)
            x = self.activation2(x)
            x = self.dropout2(x)
            x = self.conv2(x)
        else:
            x = self.bn1(x)
            x = self.activation1(x)
            x = self.maxunpool(x)
            x = self.dropout1(x)
            x = self.conv1(x)
            x = self.bn2(x)
            x = self.activation2(x)
            x = self.dropout2(x)
            x = self.conv2(x)
        return x
