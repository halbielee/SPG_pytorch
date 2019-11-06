from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.model_zoo import load_url


__all__ = ['inception_v3', 'inception_v3_spg']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def inception_v3(pretrained=False, progress=True, **kwargs):
    kwargs['turnoff'] = True
    if pretrained:
        model = Inception3(**kwargs)
        state_dict = load_url(model_urls['inception_v3_google'], progress=progress)
        remove_layer(state_dict, 'Mixed_7')
        remove_layer(state_dict, 'AuxLogits')
        remove_layer(state_dict, 'fc.')

        model.load_state_dict(state_dict, strict=False)
        return model

    return Inception3(**kwargs)


def inception_v3_spg(pretrained=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        model = Inception3(**kwargs)
        state_dict = load_url(model_urls['inception_v3_google'], progress=progress)
        remove_layer(state_dict, 'Mixed_7')
        remove_layer(state_dict, 'AuxLogits')
        remove_layer(state_dict, 'fc.')

        model.load_state_dict(state_dict, strict=False)
        return model

    return Inception3(**kwargs)


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, turnoff=False, drop_thr=((0.7, 0.05), (0.5, 0.05), (0.7, 0.1)),
                 **kwargs):

        super(Inception3, self).__init__()
        self.turnoff = turnoff
        self.drop_thr = drop_thr

        # STEM
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)

        # SPG-A / A1
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        # SPG-A / A2
        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # SPG-A / A3
        # self.SPG_A3_1a = nn.Dropout(p=0.5)
        self.SPG_A3_1b = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        # self.SPG_A3_2a = nn.Dropout(p=0.5)
        self.SPG_A3_2b = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
        )
        # self.SPG_A3_3a = nn.Dropout(p=0.5)

        # SPG-A / A4
        self.SPG_A4 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)

        # SPG-A / GAP
        self.GAP = nn.AdaptiveAvgPool2d(1)
        if not self.turnoff:
            # SPG-B
            # We do not put Sigmoid Layer to use BCEWithLogitsLoss()
            self.SPG_B_1a = nn.Sequential(
                nn.Conv2d(288, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.SPG_B_2a = nn.Sequential(
                nn.Conv2d(768, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            # shared parameter in SPG-B
            self.SPG_B_shared = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1, kernel_size=1, padding=0),
            )

            # SPG-C
            self.SPG_C = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1, kernel_size=1),
            )

        self.num_classes = num_classes
        self.interp = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        self._initialize_weights()

        # Loss Functions
        self.loss_CELoss = nn.CrossEntropyLoss()
        if not self.turnoff:
            self.loss_BCEWithLogitLoss = nn.BCEWithLogitsLoss()

        # Layers to print
        self.print_layers = dict()
        self.score = None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        """
    def forward(self, x, label=None):
        # N x 3 x 224 x 224

        # STEM
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 112 x 112
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 110 x 110
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 110 x 110
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # N x 64 x 56 x 56
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 56 x 56
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 54 x 54
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # N x 192 x 28 x 28
        # self.print_layers['layer1'] = x.clone().detach()
        # SPG-A / A1
        x = self.Mixed_5b(x)
        # N x 256 x 28 x 28
        x = self.Mixed_5c(x)
        # N x 288 x 28 x 28
        x = self.Mixed_5d(x)
        # self.print_layers['layer2'] = x.clone().detach()

        # SPG-B / B1
        if not self.turnoff:
            b1 = self.SPG_B_1a(x)
            b1 = self.SPG_B_shared(b1)

        # SPG-A / A2
        # N x 288 x 28 x 28
        x = self.Mixed_6a(x)
        # N x 768 x 28 x 28
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feat = self.Mixed_6e(x)
        # N x 768 x 28 x 28
        # self.print_layers['layer3'] = x.clone().detach()

        # SPG-B / B2
        if not self.turnoff:
            b2 = self.SPG_B_2a(x)
            b2 = self.SPG_B_shared(b2)

        # SPG-A / A3
        # x = self.SPG_A3_1a(feat)
        x = F.dropout(feat, 0.5, self.training)
        x = self.SPG_A3_1b(x)
        # x = self.SPG_A3_2a(x)
        x = F.dropout(x, 0.5, self.training)
        x = self.SPG_A3_2b(x)
        # x = self.SPG_A3_3a(x)
        x = F.dropout(x, 0.5, self.training)
        self.print_layers['layer4'] = x.clone().detach()

        # SPG-A / A4
        feat_map = self.SPG_A4(x)
        self.print_layers['feat_map'] = feat_map.clone().detach()
        # SPG-C
        if not self.turnoff:
            map_c = self.SPG_C(x)

        # SPG-A / GAP
        logit = self.GAP(feat_map)
        logit = logit.view(logit.shape[0:2])
        self.score = logit

        if self.turnoff:
            return logit

        # in case validation
        if label is None:
            _, label = torch.max(logit, dim=1)
        # attention map
        attention = self.get_attention(self.interp(feat_map), label, True)
        return [logit, b1, b2, map_c, attention]

    def get_attention(self, feat_map, label, normalize=True):
        """
        return: return attention size (batch, 1, h, w)
        """
        label = label.long()
        b = feat_map.size(0)

        # attention = torch.zeros(feat_map.size(0), 1, feat_map.size(2), feat_map.size(3))
        # attention = Variable(attention.cuda())
        # for idx in range(b):
        #     attention[idx, 0, :, :] = feat_map[idx, label.data[idx], :, :]

        attention = feat_map.detach().clone().requires_grad_(True)[range(b), label.data, :, :]
        attention = attention.unsqueeze(1)
        if normalize:
            attention = normalize_tensor(attention)
        return attention

    def get_loss(self, logits, target):
        logit, b1, b2, map_c, attention = logits

        # Loss1 / Classification Loss
        loss_cls = self.loss_CELoss(logit, target.long())

        if self.turnoff:
            return loss_cls
        # Loss2 / Attention Maps loss
        mask_attention = get_mask(attention, self.drop_thr[0][0], self.drop_thr[0][1])
        mask_b2 = get_mask(torch.sigmoid(self.interp(b2)), self.drop_thr[1][0], self.drop_thr[1][1])

        loss_b2_attention = loss_attention(self.loss_BCEWithLogitLoss, self.interp(b2).squeeze(dim=1), mask_attention)
        loss_b1_b2 = loss_attention(self.loss_BCEWithLogitLoss, self.interp(b1).squeeze(dim=1), mask_b2)

        fused_attention = (torch.sigmoid(self.interp(b1)) + torch.sigmoid(self.interp(b2))) / 2.
        mask_fused = get_mask(fused_attention.detach(), self.drop_thr[2][0], self.drop_thr[2][1])
        loss_fused_c = loss_attention(self.loss_BCEWithLogitLoss, self.interp(map_c).squeeze(dim=1), mask_fused)

        loss_total = loss_cls + loss_b2_attention + loss_b1_b2 + loss_fused_c
        if self.training:
            self.print_layers['b1'] = torch.sigmoid(self.interp(b1)).clone().detach()
            self.print_layers['b2'] = torch.sigmoid(self.interp(b2)).clone().detach()
            self.print_layers['mask_b2'] = mask_b2.clone().detach()
            self.print_layers['mask_attention'] = mask_attention.clone().detach()
            self.print_layers['fused'] = mask_fused.clone().detach()
            self.print_layers['c'] = torch.sigmoid(self.interp(map_c)).clone().detach()
        return loss_total

    def get_cam(self):
        """
        get CAM image with size (batch, class, h, w)
        """

        cam = normalize_tensor(self.print_layers['feat_map'])
        return cam, self.score

    def get_layers(self):
        return self.print_layers

def loss_attention(loss_func, logits, labels):
    pos = labels.view(-1, 1) < 255.
    return loss_func(logits.view(-1, 1)[pos], labels.view(-1, 1)[pos].detach().clone())


def get_mask(attention, thr_high=0.7, thr_low=0.05):
    # mask = torch.zeros((attention.size(0), 1, 224, 224)).fill_(255).cuda()
    mask = attention.new_zeros((attention.size(0), 1, 224, 224)).fill_(255)
    mask = mask_fg(mask, attention, thr_high)
    mask = mask_bg(mask, attention, thr_low)

    return mask


def mask_fg(mask, attention, threshold=0.5):
    """
    Fill 1.0 in the position whose value is larger than threshold
    """
    fg_val = 1.

    for i in range(attention.size(0)):
        pos_fg = attention[i] > threshold

        # calibration
        if torch.sum(pos_fg.float()).item() < 30:
            threshold = torch.max(attention[i]) * 0.7
            pos_fg = attention[i] > threshold

        # fill in the mask
        cur_mask = mask[i]
        cur_mask[pos_fg.data] = fg_val
        mask[i] = cur_mask

    return mask


def mask_bg(mask, attention, threshold=0.05):
    """
    Fill 0.0 in the position which is less than threshold
    """
    pos_bg = attention < threshold
    mask[pos_bg.data] = 0.
    return mask


def normalize_tensor(x):
    map_size = x.size()
    aggregated = x.view(map_size[0], map_size[1], -1)
    minimum, _ = torch.min(aggregated, dim=-1, keepdim=True)
    maximum, _ = torch.max(aggregated, dim=-1, keepdim=True)
    normalized = torch.div(aggregated - minimum, maximum - minimum)
    normalized = normalized.view(map_size)

    return normalized


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384,
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
