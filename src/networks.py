import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models._utils import IntermediateLayerGetter

class EAST(nn.Module):
    def __init__(self, geometry="rbox"):
        super().__init__()


        vgg16_pretrained = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
        self.vgg = vgg16_pretrained.features
        return_layers = {'9': 'pooling-2', '16': 'pooling-3', '23': 'pooling-4', '30': 'pooling-5'}
        self.features_stem = IntermediateLayerGetter(self.vgg, return_layers=return_layers)

        # self.unpool = nn.MaxUnpool2d(kernel_size=2)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1_128 = nn.Conv2d(in_channels=1024,out_channels=128,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv3_128 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv1_64 = nn.Conv2d(in_channels=384,out_channels=64,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3_64 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv1_32 = nn.Conv2d(in_channels=192,out_channels=32,kernel_size=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv3_32 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)

        self.final_ftr_conv = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm2d(32)

        self.score_conv = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)
    
        self.geometry = geometry
        if self.geometry == "rbox":
            self.text_box_conv = nn.Conv2d(in_channels=32,out_channels=4,kernel_size=1)
            self.text_rotation_conv = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)
        else:
            # quad
            self.quad_conv = nn.Conv2d(in_channels=32,out_channels=8,kernel_size=1)

        # for param in self.stem.parameters():
        #     param.requires_grad = False
        
    def forward(self, x):
        """
        Args
            x: tensor of images with shape (batch, channels, height, width)
        
        Returns
            ...
        """
        # Feature Extractor Stem
        features_dict = self.features_stem(x)
        f1 = features_dict['pooling-5']
        f2 = features_dict['pooling-4']
        f3 = features_dict['pooling-3']
        f4 = features_dict['pooling-2']

        # Feature-merging branch
        h1 = f1
        g1 = self.unpool(h1)
        h2 = torch.cat([g1, f2], dim=1)
        h2 = self.conv1_128(h2)
        h2 = self.bn1(h2)
        h2 = self.relu(h2)
        h2 = self.conv3_128(h2)
        h2 = self.bn2(h2)
        h2 = self.relu(h2)

        g2 = self.unpool(h2)
        h3 = torch.cat([g2, f3], dim=1)
        h3 = self.conv1_64(h3)
        h3 = self.bn3(h3)
        h3 = self.relu(h3)
        h3 = self.conv3_64(h3)
        h3 = self.bn4(h3)
        h3 = self.relu(h3)

        g3 = self.unpool(h3)
        h4 = torch.cat([g3, f4], dim=1)
        h4 = self.conv1_32(h4)
        h4 = self.bn5(h4)
        h4 = self.relu(h4)
        h4 = self.conv3_32(h4)
        h4 = self.bn6(h4)
        h4 = self.relu(h4)

        feature_out = self.final_ftr_conv(h4)
        feature_out = self.bn7(feature_out)
        feature_out = self.relu(feature_out)

        # Output layer
        score = self.score_conv(feature_out)

        # add sigmoid activations?
        if self.geometry == "rbox":
            text_box = self.text_box_conv(feature_out)
            text_rotation = self.text_rotation_conv(feature_out)
            geo = torch.cat([text_box, text_rotation], dim=1)
        else: # quad
            geo = self.quad_conv(feature_out)

        return score, geo

class ViTSTR(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        pass