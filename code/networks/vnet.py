import torch
from torch import nn
from .deeplab_utils import Decoder_Attention

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out

    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

"""
    以下都是复制过来的东西
"""
class InterSampleAttention(torch.nn.Module):
    """
        Implementation for inter-sample self-attention
        input size for the encoder_layers: [batch, h x w x d, dim]
    """
    def __init__(self, input_dim=256, hidden_dim=1024):
        super(InterSampleAttention, self).__init__()
        self.input_dim = input_dim
        self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4, hidden_dim, 0.5)
        # self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4 * 4, hidden_dim, 0.5)

    def forward(self, feature):
        if self.training:
            b, c, h, w, d = feature.shape
            feature = feature.permute(0, 2, 3, 4, 1).contiguous()
            feature = feature.view(b, h * w * d, c)
            feature = self.encoder_layers(feature)
            feature = feature.view(b, h, w, d, c)
            feature = feature.permute(0, 4, 1, 2, 3).contiguous()
        return feature

class IntraSampleAttention(torch.nn.Module):
    """
    Implementation for intra-sample self-attention
    input size for the encoder_layers: [h x w x d, batch, dim]
    """
    def __init__(self, input_dim=256, hidden_dim=1024):
        super(IntraSampleAttention, self).__init__()
        self.input_dim = input_dim
        # self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4, hidden_dim, 0.5)
        self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4 * 4, hidden_dim, 0.5)

    def forward(self, feature):
        if self.training:
            b, c, h, w, d = feature.shape
            feature = feature.permute(0, 2, 3, 4, 1).contiguous()
            feature = feature.view(b, h * w * d, c)
            feature = feature.permute(1, 0, 2).contiguous()
            feature = self.encoder_layers(feature)
            feature = feature.permute(1, 0, 2).contiguous()
            feature = feature.view(b, h, w, d, c)
            feature = feature.permute(0, 4, 1, 2, 3).contiguous()
        return feature

class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input, en=[]):
        if len(en) != 0:
            x1 = self.block_one(input)
            x1 = x1 + en[4]
            x1_dw = self.block_one_dw(x1)

            x2 = self.block_two(x1_dw)
            x2 = x2 + en[3]
            x2_dw = self.block_two_dw(x2)

            x3 = self.block_three(x2_dw)
            x3 = x3 + en[2]
            x3_dw = self.block_three_dw(x3)

            x4 = self.block_four(x3_dw)
            x4 = x4 + en[1]
            x4_dw = self.block_four_dw(x4)

            x5 = self.block_five(x4_dw)
            x5 = x5 + en[0]  # for 5% data

            if self.has_dropout:
                x5 = self.dropout(x5)

        else:
            x1 = self.block_one(input)
            x1_dw = self.block_one_dw(x1)

            x2 = self.block_two(x1_dw)
            x2_dw = self.block_two_dw(x2)

            x3 = self.block_three(x2_dw)
            x3_dw = self.block_three_dw(x3)

            x4 = self.block_four(x3_dw)
            x4_dw = self.block_four_dw(x4)

            x5 = self.block_five(x4_dw)

            if self.has_dropout:
                x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res

class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        # has_dropout=True
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features, with_feature=False):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # has_dropout=True
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        if with_feature:
            return out_seg, x9
        else:
            return out_seg

class DecoderAuxiliary(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(DecoderAuxiliary, self).__init__()
        # has_dropout=True
        self.has_dropout = has_dropout
        self.insert_idx = 6
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.intra_attention = IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
        self.inter_attention = InterSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)

    def get_dim(self, idx):
        if idx == 6:
            # print("self.block_eight.conv.shape ", self.block_eight.conv.shape)
            return self.block_six.conv[3].weight.shape[0]

    def forward(self, features, with_feature=False):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        if self.insert_idx == 6:
            x6 = self.intra_attention(x6)
            x6 = self.inter_attention(x6)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # has_dropout=True
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        if with_feature:
            return out_seg, x9
        else:
            return out_seg
class DecoderAuxiliary_intra(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(DecoderAuxiliary_intra, self).__init__()
        # has_dropout=True
        self.has_dropout = has_dropout
        self.insert_idx = 6
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.intra_attention = IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
        # self.inter_attention = InterSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)

    def get_dim(self, idx):
        if idx == 6:
            # print("self.block_four.conv[6].weight.shape[0] ", self.block_four.conv[6].weight.shape[0])
            return self.block_six.conv[3].weight.shape[0]

    def forward(self, features, with_feature=False):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        if self.insert_idx == 6:
            x6 = self.intra_attention(x6)
            # x6 = self.inter_attention(x6)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # has_dropout=True
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        if with_feature:
            return out_seg, x9
        else:
            return out_seg
class DecoderAuxiliary_inter(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(DecoderAuxiliary_inter, self).__init__()
        # has_dropout=True
        self.has_dropout = has_dropout
        self.insert_idx = 6
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.intra_attention = IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
        self.inter_attention = InterSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)

    def get_dim(self, idx):
        if idx == 6:
            # print("self.block_four.conv[6].weight.shape[0] ", self.block_four.conv[6].weight.shape[0])
            return self.block_six.conv[3].weight.shape[0]

    def forward(self, features, with_feature=False):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        if self.insert_idx == 6:
            # x6 = self.intra_attention(x6)
            x6 = self.inter_attention(x6)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # has_dropout=True
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        if with_feature:
            return out_seg, x9
        else:
            return out_seg
class DecoderAuxiliary_inter_attention(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0, attention_mode='CBAM'):
        super(DecoderAuxiliary_inter_attention, self).__init__()
        # has_dropout=True
        self.has_dropout = has_dropout
        self.insert_idx = 6
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        """三个注意力块"""
        if attention_mode == 'CBAM':
            # self.decoder_attention = Decoder_Attention(n_classes, 256, attention_mode='CBAM')
            # self.decoder_attention = IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
            self.decoder_attention = nn.Sequential(IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
                                                   , InterSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4))
        elif attention_mode == 'SA':
            self.decoder_attention = Decoder_Attention(n_classes, 256, attention_mode='SA')
        else:
            self.decoder_attention = Decoder_Attention(n_classes, 256, attention_mode='CA')
        # self.intra_attention = IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
        # self.inter_attention = InterSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)

    def get_dim(self, idx):
        if idx == 6:
            # print("self.block_four.conv[6].weight.shape[0] ", self.block_four.conv[6].weight.shape[0])
            return self.block_six.conv[3].weight.shape[0]

    def forward(self, features, with_feature=False):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up_ori = self.block_five_up(x5)
        x5_up = x5_up_ori + x4

        x6 = self.block_six(x5_up)
        if self.insert_idx == 6:
            # x6 = self.intra_attention(x6)
            # x6 torch.Size([4, 128, 14, 14, 10])
            # print("\n#######################  x6.shape #######################: ", x6.shape)
            x6 = self.decoder_attention(x6)
            # x6 = self.inter_attention(x6)
        x6_up_ori = self.block_six_up(x6)
        x6_up = x6_up_ori + x3

        x7 = self.block_seven(x6_up)
        x7_up_ori = self.block_seven_up(x7)
        x7_up = x7_up_ori + x2

        x8 = self.block_eight(x7_up)
        x8_up_ori = self.block_eight_up(x8)
        x8_up = x8_up_ori + x1
        x9 = self.block_nine(x8_up)
        # has_dropout=True
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        if with_feature:
            return out_seg, [x5, x5_up_ori, x6_up_ori, x7_up_ori, x8_up_ori]
        else:
            return out_seg
class EncoderAuxiliary(nn.Module):
    """
    encoder for auxiliary model with CMA
    """
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4):
        super(EncoderAuxiliary, self).__init__()
        self.insert_idx = insert_idx
        self.cma_type = cma_type

        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # print(self.get_dim(self.insert_idx))
        if self.cma_type == 'v2+':  # True
            self.intra_attention = IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
        self.inter_attention = InterSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)

    def get_dim(self, idx):
        if idx == 4:
            # print("self.block_four.conv[6].weight.shape[0] ", self.block_four.conv[6].weight.shape[0])
            return self.block_four.conv[6].weight.shape[0]

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        # cma layers
        if self.insert_idx == 4:
            if self.cma_type == "v2+":
                # print("\nintra_attention's input shape is ", x4.shape)
                x4 = self.intra_attention(x4)
                # print("\nintra_attention's output shape is ", x4.shape)
            x4 = self.inter_attention(x4)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res
class EncoderAuxiliary_intra(nn.Module):
    """
    encoder for auxiliary model with CMA
    """
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4):
        super(EncoderAuxiliary_intra, self).__init__()
        self.insert_idx = insert_idx
        self.cma_type = cma_type

        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # print(self.get_dim(self.insert_idx))
        if self.cma_type == 'v2+':  # True
            self.intra_attention = IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
        # self.inter_attention = InterSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)

    def get_dim(self, idx):
        if idx == 4:
            return self.block_four.conv[6].weight.shape[0]

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        # cma layers
        if self.insert_idx == 4:
            if self.cma_type == "v2+":
                x4 = self.intra_attention(x4)
            # x4 = self.inter_attention(x4)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res
class EncoderAuxiliary_inter(nn.Module):
    """
    encoder for auxiliary model with CMA
    """
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4):
        super(EncoderAuxiliary_inter, self).__init__()
        self.insert_idx = insert_idx
        self.cma_type = cma_type

        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # print(self.get_dim(self.insert_idx))
        # if self.cma_type == 'v2+':  # True
        #     self.intra_attention = IntraSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)
        self.inter_attention = InterSampleAttention(self.get_dim(self.insert_idx), self.get_dim(self.insert_idx) * 4)

    def get_dim(self, idx):
        if idx == 4:
            return self.block_four.conv[6].weight.shape[0]

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        # cma layers
        if self.insert_idx == 4:
            # if self.cma_type == "v2+":
            #     x4 = self.intra_attention(x4)
            x4 = self.inter_attention(x4)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res

class CAML3d_v2_MTNet_CDMA_DN(nn.Module):
    """
    Use CMA on Encoder layer 4
    With different upsample 两个 Encoder 和 Decoder 表示图中的两个模型
    """
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4, feat_dim=32):
        super(CAML3d_v2_MTNet_CDMA_DN, self).__init__()
        self.has_dropout = has_dropout
        self.cma_type = cma_type
        self.insert_idx = insert_idx
        assert self.insert_idx == 4
        self.encoder1 = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        # has_dropout=True
        # self.decoder1 = DecoderAuxiliary(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        # self.decoder2 = DecoderAuxiliary_intra(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder1 = DecoderAuxiliary_inter_attention(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1, attention_mode='CBAM')
        self.decoder2 = DecoderAuxiliary_inter_attention(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1, attention_mode='SA')
        self.decoder3 = DecoderAuxiliary_inter_attention(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1, attention_mode='CA')
        from dsbn import DomainSpecificBatchNorm2d
        # self.dual_batch_norm = DomainSpecificBatchNorm2d(num_features=112, num_domains=2, momentum=0.1)
        self.dual_batch_norm = DomainSpecificBatchNorm2d(num_features=1, num_domains=2, momentum=0.1)
        self.dual_batch_norm_2 = DomainSpecificBatchNorm2d(num_features=1, num_domains=2, momentum=0.1)
        self.dual_batch_norm_3 = DomainSpecificBatchNorm2d(num_features=1, num_domains=2, momentum=0.1)

    def forward(self, input, en=[]):
        features1 = self.encoder1(input, en)

        out_seg1, stage_feat1 = self.decoder1(features1, with_feature=True)
        out_seg2, stage_feat2 = self.decoder2(features1, with_feature=True)
        out_seg3, stage_feat3 = self.decoder3(features1, with_feature=True)
        if self.training:  # 训练模式
            return out_seg1, out_seg2, out_seg3, [stage_feat1, stage_feat3, stage_feat2]
        else:
            # return out_seg1
            # return out_seg2
            # return out_seg3
            return (out_seg1 + out_seg2 + out_seg3) / 3

class CAML3d_v2_MTNet_DN(nn.Module):
    """
    Use CMA on Encoder layer 4
    With different upsample 两个 Encoder 和 Decoder 表示图中的两个模型
    """
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4, feat_dim=32):
        super(CAML3d_v2_MTNet_DN, self).__init__()
        self.has_dropout = has_dropout
        self.cma_type = cma_type
        self.insert_idx = insert_idx
        assert self.insert_idx == 4
        self.encoder1 = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        # self.encoder1 = EncoderAuxiliary_intra(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        # self.encoder1 = EncoderAuxiliary(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        # has_dropout=True
        self.decoder1 = DecoderAuxiliary(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        # self.decoder2 = DecoderAuxiliary_intra(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder2 = DecoderAuxiliary_intra(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        # self.decoder3 = DecoderAuxiliary_inter(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = DecoderAuxiliary_inter(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        # self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        from dsbn import DomainSpecificBatchNorm2d
        # self.dual_batch_norm = DomainSpecificBatchNorm2d(num_features=112, num_domains=2, momentum=0.1)
        self.dual_batch_norm = DomainSpecificBatchNorm2d(num_features=1, num_domains=2, momentum=0.1)
        self.dual_batch_norm_2 = DomainSpecificBatchNorm2d(num_features=1, num_domains=2, momentum=0.1)
        self.dual_batch_norm_3 = DomainSpecificBatchNorm2d(num_features=1, num_domains=2, momentum=0.1)

    def forward(self, input):
        features1 = self.encoder1(input)

        out_seg1, _ = self.decoder1(features1, with_feature=True)
        out_seg2, _ = self.decoder2(features1, with_feature=True)
        out_seg3, _ = self.decoder3(features1, with_feature=True)
        if(self.has_dropout):  # 训练模式
            return out_seg1, out_seg2, out_seg3
        else:
            # return out_seg1
            # return out_seg2
            return out_seg3

class CAML3d_v2_MTNet(nn.Module):
    """
    Use CMA on Encoder layer 4
    With different upsample 两个 Encoder 和 Decoder 表示图中的两个模型
    """
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4, feat_dim=32):
        super(CAML3d_v2_MTNet, self).__init__()
        self.has_dropout = has_dropout
        self.cma_type = cma_type
        self.insert_idx = insert_idx
        assert self.insert_idx == 4
        self.encoder1 = EncoderAuxiliary_intra(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        # self.encoder1 = EncoderAuxiliary(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        # has_dropout=True
        self.decoder1 = DecoderAuxiliary(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder2 = DecoderAuxiliary_intra(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = DecoderAuxiliary_inter(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        # self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)

    def forward(self, input):
        features1 = self.encoder1(input)

        out_seg1, _ = self.decoder1(features1, with_feature=True)
        out_seg2, _ = self.decoder2(features1, with_feature=True)
        out_seg3, _ = self.decoder3(features1, with_feature=True)
        if(self.has_dropout):  # 训练模式
            return out_seg1, out_seg2, out_seg3
        else:
            # return out_seg1
            return out_seg3

class CAML3d_v1(nn.Module):
    """
    Use CMA on Encoder layer 4
    With different upsample 两个 Encoder 和 Decoder 表示图中的两个模型
    """

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, cma_type='v2+', insert_idx=4, feat_dim=32):
        super(CAML3d_v1, self).__init__()
        self.cma_type = cma_type
        self.insert_idx = insert_idx
        assert self.insert_idx == 4
        # self.encoder1 = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.encoder2 = EncoderAuxiliary(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,
                                  cma_type=self.cma_type, insert_idx=self.insert_idx)
        # has_dropout=True
        # self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.projection_head1 = nn.Sequential(
            nn.Linear(n_filters, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head1 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.projection_head2 = nn.Sequential(
            nn.Linear(n_filters, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head2 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, input):
        # features1 = self.encoder1(input)
        features2 = self.encoder2(input)
        # out_seg1, embedding1 = self.decoder1(features1, with_feature=True)
        out_seg2, embedding2 = self.decoder2(features2, with_feature=True)
        # return out_seg1, out_seg2, embedding1, embedding2
        return out_seg2, embedding2
