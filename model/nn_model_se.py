import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

KERNEL_SIZE = 3
STRIDE = 1
LAST_DIM = 8
NUM_CLASSES = 10


# -----------------------------
# Utility: 参数初始化
# -----------------------------
def init_params(models):
    if not isinstance(models, list):
        models = [models]

    for model in models:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# -----------------------------
# SE Block (Squeeze-and-Excitation)
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


# -----------------------------
# BasicBlock for decoder
# -----------------------------
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, num_feature_map=64):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, KERNEL_SIZE, STRIDE, padding="same")
        self.conv1_bn = nn.BatchNorm2d(c_out)
        self.conv11 = nn.Conv2d(c_out, num_feature_map, KERNEL_SIZE, STRIDE, padding="same")
        self.conv11_bn = nn.BatchNorm2d(num_feature_map)
        self.conv12 = nn.Conv2d(num_feature_map, c_out, KERNEL_SIZE, STRIDE, padding="same")
        self.conv12_bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.conv1_bn(self.conv1(x))
        x = nn.ReLU()(x)
        xi = x
        x = self.conv11_bn(self.conv11(x))
        x = nn.ReLU()(x)
        x = self.conv12_bn(self.conv12(x))
        x = x + xi
        return x


# -----------------------------
# Encoder with SE attention
# -----------------------------
class ResNetEnc(nn.Module):
    def __init__(self, image_size=32):
        super(ResNetEnc, self).__init__()
        self.image_size = image_size
        self.model = resnet50(pretrained=True)

        # 替换输入层，适配 CIFAR-10 (32x32)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        if self.image_size == 32:
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1, bias=False
            )
            delattr(self.model, "maxpool")
            init_params(self.model.conv1)

        # 在每层后添加 SE 注意力模块
        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.layer1(x)
        x = self.se1(x)
        x = self.model.layer2(x)
        x = self.se2(x)
        x = self.model.layer3(x)
        x = self.se3(x)
        x = self.model.layer4(x)
        x = self.se4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# -----------------------------
# ResNetVAE with SE-enhanced Encoder
# -----------------------------
class ResNetVAE(nn.Module):
    def __init__(self, REncoder, latent_dim=256, latent_expand_ratio=8, decoder_layers=4):
        super(ResNetVAE, self).__init__()

        self.encoder_module = nn.ModuleList([REncoder])
        self.decoder_module = nn.ModuleList()
        for _ in range(decoder_layers):
            self.decoder_module.append(BasicBlock(latent_dim, latent_dim))

        self.output_layer = nn.ModuleList(
            [BasicBlock(latent_dim, LAST_DIM), BasicBlock(LAST_DIM, 3)]
        )
        self.output_layer2 = nn.Linear(latent_dim, NUM_CLASSES)
        self.mean_val = nn.Linear(latent_dim * latent_expand_ratio, latent_dim)
        self.var_val = nn.Linear(latent_dim * latent_expand_ratio, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x, deterministic=True, classification_only=True):
        for i_op in self.encoder_module:
            x = i_op(x)
            x = nn.ReLU()(x)

        mu = self.mean_val(x)
        logvar = self.var_val(x)

        if deterministic:
            sample_z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sample_z = eps * std + mu

        x = sample_z[:, :, None, None]
        z = x
        y = self.output_layer2(torch.mean(z, dim=(2, 3)))

        x = nn.Upsample(scale_factor=2, mode="nearest")(x)
        for i_op in self.decoder_module:
            x = i_op(x)
            x = nn.Upsample(scale_factor=2, mode="nearest")(x)

        for i_op in self.output_layer:
            x = i_op(x)

        x = torch.tanh(x)
        if classification_only:
            return y
        else:
            return x, z, y, mu, logvar
