""" Full assembly of the parts to form the complete network """

from .unet_parts import *


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits



# import torch
# # import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# # import torch.optim as optim
# # from torch.autograd import Variable
# # from tqdm import trange
# # from time import sleep
# use_gpu = torch.cuda.is_available()


# class UNet(nn.Module):
#     def contracting_block(self, in_channels, out_channels, kernel_size=3):
#         block = torch.nn.Sequential(
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
#                     torch.nn.ReLU(),
#                     torch.nn.Dropout2d(),
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
#                     torch.nn.ReLU(),
#                     torch.nn.BatchNorm2d(out_channels),
#                 )
#         return block
    
#     def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
#             block = torch.nn.Sequential(
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
#                     torch.nn.ReLU(),
#                     torch.nn.Dropout2d(),
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
#                     torch.nn.ReLU(),
#                     torch.nn.BatchNorm2d(mid_channel),
#                     torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
#                     )
#             return  block
    
#     def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
#             block = torch.nn.Sequential(
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
#                     torch.nn.ReLU(),
#                     torch.nn.Dropout2d(),
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
#                     torch.nn.ReLU(),
#                     torch.nn.BatchNorm2d(mid_channel),
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
#                     torch.nn.Sigmoid(),
#                     )
#             return  block
    
#     def __init__(self, in_channel, out_channel):
#         super(UNet, self).__init__()
#         #Encode
#         self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
#         self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv_encode2 = self.contracting_block(64, 128)
#         self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv_encode3 = self.contracting_block(128, 256)
#         self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv_encode4 = self.contracting_block(256, 512)
#         self.conv_maxpool4 = torch.nn.MaxPool2d(kernel_size=2)
#         # Bottleneck
#         self.bottleneck = torch.nn.Sequential(
#                             torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024, padding=1),
#                             torch.nn.ReLU(),
#                             torch.nn.BatchNorm2d(1024),
#                             torch.nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024, padding=1),
#                             torch.nn.ReLU(),
#                             torch.nn.BatchNorm2d(1024),
#                             torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
#                             )
#         # Decode
#         self.conv_decode4 = self.expansive_block(1024, 512, 256)
#         self.conv_decode3 = self.expansive_block(512, 256, 128)
#         self.conv_decode2 = self.expansive_block(256, 128, 64)
#         self.final_layer = self.final_block(128, 64, out_channel)
        
#     def crop_and_concat(self, upsampled, bypass, crop=False):
#         if crop:
#             c = (bypass.size()[2] - upsampled.size()[2]) // 2
#             bypass = F.pad(bypass, (-c, -c, -c, -c))
#         return torch.cat((upsampled, bypass), 1)
    
#     def forward(self, x):
#         # Encode
#         encode_block1 = self.conv_encode1(x)
#         encode_pool1 = self.conv_maxpool1(encode_block1)
#         #print(encode_block1.shape)
#         encode_block2 = self.conv_encode2(encode_pool1)
#         encode_pool2 = self.conv_maxpool2(encode_block2)
#         encode_block3 = self.conv_encode3(encode_pool2)
#         encode_pool3 = self.conv_maxpool3(encode_block3)
#         encode_block4 = self.conv_encode4(encode_pool3)
#         encode_pool4  = self.conv_maxpool4(encode_block4)
#         # Bottleneck
#         bottleneck1 = self.bottleneck(encode_pool4)
#         # Decode
#         #print(x.shape, encode_block1.shape, encode_block2.shape, encode_block3.shape, encode_pool3.shape, bottleneck1.shape)
#         #print('Decode Block 3')
#         #print(bottleneck1.shape, encode_block3.shape)
#         decode_block4 = self.crop_and_concat(bottleneck1, encode_block4, crop=True)
#         cat_layer3 = self.conv_decode4(decode_block4)
#         decode_block3 = self.crop_and_concat(cat_layer3, encode_block3, crop=True)
#         #print(decode_block3.shape)
#         #print('Decode Block 2')
#         cat_layer2 = self.conv_decode3(decode_block3)
#         #print(cat_layer2.shape, encode_block2.shape)
#         decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
#         cat_layer1 = self.conv_decode2(decode_block2)
#         #print(cat_layer1.shape, encode_block1.shape)
#         #print('Final Layer')
#         #print(cat_layer1.shape, encode_block1.shape)
#         decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
#         #print(decode_block1.shape)
#         final_layer = self.final_layer(decode_block1)
#         #print(final_layer.shape)
#         return  final_layer





import torch.nn as nn
from torch.nn.modules.loss import _Loss


class SoftDiceLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, y_pred, y_gt):
        numerator = torch.sum(y_pred*y_gt)
        denominator = torch.sum(y_pred*y_pred + y_gt*y_gt)
        return numerator/denominator


class First2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder2D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=downsample_kernel),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1)
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class First3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder3D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder3D, self).__init__()

        layers = [
            nn.MaxPool3d(kernel_size=downsample_kernel),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center3D, self).__init__()

        layers = [
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1)
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .blocks import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last2D(conv_depths[1], conv_depths[0], out_channels))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec