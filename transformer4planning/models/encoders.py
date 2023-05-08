import torch
from torch import nn, Tensor
import numpy as np

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, config):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.ReLU()
        )  # 14 x 14
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU()
        )  # 28 x 28
        self.double_conv3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU()
        )  # 56 x 56
        self.double_conv4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )  # 112 x 112

        self.double_conv5 = nn.Sequential(
            nn.Conv2d(128, config.d_embed, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.d_embed, momentum=1, affine=True),
            nn.ReLU(),
        )  # 224 x 224

    def forward(self, x, concat_features):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.upsample(out)  # block 1
        out = torch.cat((out, concat_features[-1]), dim=1)
        out = self.double_conv1(out)
        out = self.upsample(out)  # block 2
        out = torch.cat((out, concat_features[-2]), dim=1)
        out = self.double_conv2(out)
        out = self.upsample(out)  # block 3
        out = torch.cat((out, concat_features[-3]), dim=1)
        out = self.double_conv3(out)
        out = self.upsample(out)  # block 4
        out = torch.cat((out, concat_features[-4]), dim=1)
        out = self.double_conv4(out)
        # return out
        out = self.upsample(out)  # block 5
        out = torch.cat((out, concat_features[-5]), dim=1)
        out = self.double_conv5(out)
        return out


class CNNDownSamplingResNet18(nn.Module):
    def __init__(self, d_embed, in_channels):
        super(CNNDownSamplingResNet18, self).__init__()
        import torchvision.models as models
        self.cnn = models.resnet18(pretrained=False, num_classes=d_embed)
        self.cnn = torch.nn.Sequential(*(list(self.cnn.children())[1:-1]))
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=d_embed, bias=True)
        )
        # self.cnn = models.vgg11(pretrained=False, num_classes=config.d_embed)
        # self.cnn.features = self.cnn.features[1:]
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # )

    def forward(self, x):
        x = self.layer1(x)
        x = self.cnn(x)
        output = self.classifier(x.squeeze(-1).squeeze(-1))
        return output


class CNNDownSampling(nn.Module):
    def __init__(self, config, in_channels):
        super(CNNDownSampling, self).__init__()
        import torchvision.models as models
        self.cnn = models.vgg16(pretrained=False, num_classes=config.d_embed)
        self.cnn.features = self.cnn.features[1:]
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        output = self.cnn(x)
        assert output.shape == (len(x), args.hidden_size), output.shape
        return output


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self, config, in_channels):
        super(CNNEncoder, self).__init__()
        import torchvision.models as models
        features = list(models.vgg16_bn(pretrained=False).features)
        # in_channels = 101
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        )
        self.features = nn.ModuleList(features)[1:]  # .eval()
        # print (nn.Sequential(*list(models.vgg16_bn(pretrained=True).children())[0]))
        # self.features = nn.ModuleList(features).eval()
        self.decoder = RelationNetwork(config)

    def forward(self, x):
        results = []
        x = self.layer1(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 11, 21, 31, 41}:
                results.append(x)

        output = self.decoder(x, results)
        output = output.permute(0, 2, 3, 1)
        # assert output.shape == (len(x), 224, 224, config.d_embed), output.shape
        return output
