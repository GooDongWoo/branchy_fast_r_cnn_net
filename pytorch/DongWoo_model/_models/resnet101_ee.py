#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/resnet.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PyTorch implementation of the ResNet backbone for use as a feature extractor
# in Faster R-CNN. See the Backbone base class for a description of how the
# classes here are structured.
#
# References
# ----------
# [1] "Deep Residual Learning for Image Recognition"
#     Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#

# from enum import Enums
import sys
import os

from math import ceil
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from _datasets import image
from .backbone import Backbone
from .myResNet import ResNet_2EE, BottleNeck, get_output_shape, resnet101
from torchsummary import summary

device = 'cuda'


# class Architecture(Enum):
#   ResNet50 = "ResNet50"
#   ResNet101 = "ResNet101"
#   ResNet152 = "ResNet152"


class FeatureExtractor(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self._feature_extractor1 = nn.Sequential(
            resnet.conv1,  # 0
            resnet.bn1,  # 1
            resnet.relu,  # 2
            resnet.maxpool  # 3
        )
        self._feature_extractor2 = nn.Sequential(
            resnet.layer1
        )

        # create exit by FE(Feature Extractor)

        # Freeze initial layers
        self._freeze(resnet.conv1)
        self._freeze(resnet.bn1)
        self._freeze(resnet.layer1)

        # Ensure that all batchnorm layers are frozen, as described in Appendix A
        # of [1]
        self._freeze_batchnorm(self._feature_extractor1)
        self._freeze_batchnorm(self._feature_extractor2)

    # Override nn.Module.train()
    def train(self, mode=True):
        super().train(mode)

        #
        # During training, set all frozen blocks to evaluation mode and ensure that
        # all the batchnorm layers are also in evaluation mode. This is extremely
        # important and neglecting to do this will result in severely degraded
        # training performance.
        #
        if mode:
            # Set fixed blocks to be in eval mode
            if mode:
                # Set fixed blocks to be in eval mode
                self._feature_extractor1.eval()
                self._feature_extractor2.eval()
                # self._feature_extractor2[1].train()

                # *All* batchnorm layers in eval mode
                def set_bn_eval(module):
                    if type(module) == nn.BatchNorm2d:
                        module.eval()

                self._feature_extractor1.apply(set_bn_eval)
                self._feature_extractor2.apply(set_bn_eval)

    def forward(self, image_data):
        y = self._feature_extractor1(image_data)
        y = self._feature_extractor2(y)
        return y

    @staticmethod
    def _freeze(layer):
        for name, parameter in layer.named_parameters():
            parameter.requires_grad = False

    def _freeze_batchnorm(self, block):
        for child in block.modules():
            if type(child) == nn.BatchNorm2d:
                self._freeze(layer=child)


class PoolToFeatureVector(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self._layer2 = resnet.layer2
        self._layer3 = resnet.layer3
        self._layer4 = resnet.layer4
        self._freeze_batchnorm(self._layer4)

    def train(self, mode=True):
        # See comments in FeatureVector.train()
        super().train(mode)
        if mode:
            def set_bn_eval(module):
                if type(module) == nn.BatchNorm2d:
                    module.eval()

            self._layer4.apply(set_bn_eval)

    def forward(self, rois):
        rois = self._layer2(rois)
        rois = self._layer3(rois)
        y = self._layer4(rois)  # (N, 1024, 7, 7) -> (N, 2048, 4, 4)

        # Average together the last two dimensions to remove them -> (N, 2048).
        # It is also possible to max pool, e.g.:
        # y = F.adaptive_max_pool2d(y, output_size = 1).squeeze()
        # This may even be better (74.96% mAP for ResNet50 vs. 73.2% using the
        # current method).
        y = y.mean(-1).mean(-1)  # use mean to remove last two dimensions -> (N, 2048)
        return y

    @staticmethod
    def _freeze(layer):
        for name, parameter in layer.named_parameters():
            parameter.requires_grad = False

    def _freeze_batchnorm(self, block):
        for child in block.modules():
            if type(child) == nn.BatchNorm2d:
                self._freeze(layer=child)


class ResNetBackbone(Backbone):
    def __init__(self, architecture):
        super().__init__()

        # Backbone properties. Image preprocessing parameters are common to all
        # Torchvision ResNet models and are described in the documentation, e.g.,
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
        self.feature_map_channels = 256  # feature extractor output channels
        self.feature_pixels = 4  # ResNet feature maps are 1/16th of the original image size, similar to VGG-16 feature extractor
        self.feature_vector_size = 2048  # linear feature vector size after pooling
        self.image_preprocessing_params = image.PreprocessingParams(channel_order=image.ChannelOrder.RGB,
                                                                    scaling=1.0 / 255.0, means=[0.485, 0.456, 0.406],
                                                                    stds=[0.229, 0.224, 0.225])

        # Construct model and pre-load with ImageNet weights
        if architecture == "ResNet101":
            script_dir = os.path.dirname(__file__)
            pth_rel_path = "pretrained_model"
            file_name = "pretrained_resnet.pth"
            abs_file_path = os.path.join(script_dir, pth_rel_path)
            try:
                if not os.path.exists(abs_file_path):
                    os.makedirs(abs_file_path)
            except OSError:
                print('Error: Creating directory. ' + abs_file_path)
            abs_file_path = os.path.join(abs_file_path, file_name)
            resnet_a = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
            # t.save(resnet_a.state_dict(), abs_file_path)

            # summary(resnet, (3, 32, 32), device='cuda')

        else:
            raise ValueError("Invalid ResNet architecture value: %s" % architecture.value)
        # print("Loaded IMAGENET1K_V1 pre-trained weights for Torchvision %s backbone" % architecture.value)

        # Feature extractor: given image data of shape (batch_size, channels, height, width),
        # produces a feature map of shape (batch_size, 1024, ceil(height/16), ceil(width/16))
        self.feature_extractor = FeatureExtractor(resnet=resnet_a)

        # Conversion of pooled features to head input
        self.pool_to_feature_vector = PoolToFeatureVector(resnet=resnet_a)

    def compute_feature_map_shape(self, image_shape):
        """
        Computes feature map shape given input image shape. Unlike VGG-16, ResNet
        convolutional layers use padding and the resultant dimensions are therefore
        not simply an integral division by 16. The calculation here works well
        enough but it is not guaranteed that the simple conversion of feature map
        coordinates to input image pixel coordinates in anchors.py is absolutely
        correct.

        Parameters
        ----------
        image_shape : Tuple[int, int, int]
          Shape of the input image, (channels, height, width). Only the last two
          dimensions are relevant, allowing image_shape to be either the shape
          of a single image or the entire batch.

        Returns
        -------
        Tuple[int, int, int]
          Shape of the feature map produced by the feature extractor,
          (feature_map_channels, feature_map_height, feature_map_width).
        """
        image_width = image_shape[-1]
        image_height = image_shape[-2]
        return (
        self.feature_map_channels, ceil(image_height / self.feature_pixels), ceil(image_width / self.feature_pixels))