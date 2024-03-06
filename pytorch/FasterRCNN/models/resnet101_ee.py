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

from enum import Enum
from math import ceil
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from ..datasets import image
from .backbone import Backbone
from .myResNet import ResNet_2EE, BottleNeck,  get_output_shape
from torchsummary import summary
device = 'cuda'

class Architecture(Enum):
  ResNet50 = "ResNet50"
  ResNet101 = "ResNet101"
  ResNet152 = "ResNet152"

class ConvBasic(nn.Module):
    def __init__(self, chanIn, chanOut, k=3, s=1,
                 p=1, p_ceil_mode=False, bias=False):
      super(ConvBasic, self).__init__()
      self.conv = nn.Sequential(
        nn.Conv2d(chanIn, chanOut, kernel_size=k, stride=s,
                  padding=p, bias=bias),
        nn.BatchNorm2d(chanOut),
        nn.ReLU(True)  # in place
      )

    def forward(self, x):
      return self.conv(x)

class IntrFE(nn.Module):
    # intermediate classifer head to be attached along the backbone
    # Inpsired by MSDNet classifiers (from HAPI):
    # https://github.com/kalviny/MSDNet-PyTorch/blob/master/models/msdnet.py

    def __init__(self, chanIn, input_shape, classes, bb_index, interChans=64, last_conv_chan=32):
      super(IntrFE, self).__init__()

      # index for the position in the backbone layer
      self.bb_index = bb_index
      # input shape to automatically size linear layer
      self.input_shape = input_shape

      # intermediate conv channels
      # interChans = 128 # TODO reduce size for smaller nets
      self.interChans = interChans
      self.last_conv_chan = last_conv_chan
      # conv, bnorm, relu 1
      layers = nn.ModuleList()
      self.conv1 = ConvBasic(chanIn, interChans, k=5, s=1, p=2)
      layers.append(self.conv1)
      self.conv2 = ConvBasic(interChans, interChans, k=3, s=1, p=1)
      layers.append(self.conv2)
      self.conv3 = ConvBasic(interChans, last_conv_chan, k=3, s=1, p=1)
      layers.append(self.conv3)
      self.layers = layers

      # self.linear_dim = int(torch.prod(torch.tensor(self._get_linear_size(layers))))
      # # print(f"Classif @ {self.bb_index} linear dim: {self.linear_dim}") #check linear dim
      #
      # # linear layer
      # self.linear = nn.Sequential(
      #   nn.Flatten(),
      #   nn.Linear(self.linear_dim, classes)
      # )

    def _get_linear_size(self, layers):
      for layer in layers:
        self.input_shape = get_output_shape(layer, self.input_shape)
      return self.input_shape

    def forward(self, x):
      for layer in self.layers:
        x = layer(x)
      return x

class FeatureExtractor(nn.Module):
  def __init__(self, resnet):
    super().__init__()
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu
    self.maxpool = resnet.maxpool
    self.layer1 = resnet.layer1
    self.layer2 = resnet.layer2
    self.layer3 = resnet.layer3

    # create exit by FE(Feature Extractor)
    self.exits = nn.ModuleList()
    self.exit_num = 2
    self.exit_loss_weights = [1.0, 1.0]  # for training need to match total exits_num
    self.exit_threshold = t.tensor([0.8],
                                       dtype=t.float32).to(device)  # for fast inference  #TODO: inference variable(not constant 0.8) need to make parameter
    self._build_exits(resnet)

    # Freeze initial layers
    self._freeze(self.conv1)
    self._freeze(self.bn1)
    self._freeze(self.layer1)

    # Ensure that all batchnorm layers are frozen, as described in Appendix A
    # of [1]
    self._freeze_batchnorm(self.conv1)
    self._freeze_batchnorm(self.bn1)
    self._freeze_batchnorm(self.relu)
    self._freeze_batchnorm(self.maxpool)
    self._freeze_batchnorm(self.layer1)
    self._freeze_batchnorm(self.layer2)
    self._freeze_batchnorm(self.layer3)

  # Override nn.Module.train()
  def train(self, mode = True):
    super().train(mode)

    #
    # During training, set all frozen blocks to evaluation mode and ensure that
    # all the batchnorm layers are also in evaluation mode. This is extremely
    # important and neglecting to do this will result in severely degraded
    # training performance.
    #
    if mode:
      # Set fixed blocks to be in eval mode
      self.conv1.eval()
      self.bn1.eval()
      self.relu.eval()
      self.maxpool.eval()
      self.layer1.eval()
      self.layer2.eval()
      self.layer3.eval()

      self.layer2.train()
      self.layer3.train()

      # *All* batchnorm layers in eval mode
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()
      self.conv1.apply(set_bn_eval)
      self.bn1.apply(set_bn_eval)
      self.relu.apply(set_bn_eval)
      self.maxpool.apply(set_bn_eval)
      self.layer1.apply(set_bn_eval)
      self.layer2.apply(set_bn_eval)
      self.layer3.apply(set_bn_eval)




  def exit_criterion_top1(self, x):  # NOT for batch size > 1 (in inference mode)
    with t.no_grad():
      pk = nn.functional.softmax(x, dim=-1).to('cuda')
      top1 = t.log(pk) * pk
      return (top1 < self.exit_threshold).cpu().detach().numpy()




  def forward(self, image_data):
    res  =[]
    x = self.conv1(image_data)
    x = self.bn1(x)
    x = self.relu(x)
    y = self.maxpool(x)
    res.append(self.exits[0](y))
    # if self.exit_criterion_top1(res):
    #   return res
    for b in self.layer1:
      y = b(y)
    for b in self.layer2:
      y = b(y)
    res.append(self.exits[1](y))
    # print("res : ", res.shape)
    return res[1]


  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)

  def _build_exits(self, resnet):
    # input shape ? does it have to be fixxed?
    previous_shape = get_output_shape(resnet.init_conv, resnet.input_shape)
    ee1 = IntrFE(resnet.in_chan_sizes[0],
                      # TODO: branch before conv2 ->conv_x//why before conv2 not after? beacasue author did it wtf..
                      previous_shape, resnet.num_classes, 0)
    # ee1 = nn.Sequential(self.layer3)
    self.exits.append(ee1)

    # final exit
    self.exits.append(self.layer3)


class PoolToFeatureVector(nn.Module):
  def __init__(self, resnet):
    super().__init__()
    self._layer4 = resnet.layer4
    self._freeze_batchnorm(self._layer4)

  def train(self, mode = True):
    # See comments in FeatureVector.train()
    super().train(mode)
    if mode:
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()
      self._layer4.apply(set_bn_eval)

  def forward(self, rois):
    y = self._layer4(rois)  # (N, 1024, 7, 7) -> (N, 2048, 4, 4)

    # Average together the last two dimensions to remove them -> (N, 2048).
    # It is also possible to max pool, e.g.:
    # y = F.adaptive_max_pool2d(y, output_size = 1).squeeze()
    # This may even be better (74.96% mAP for ResNet50 vs. 73.2% using the
    # current method).
    y = y.mean(-1).mean(-1) # use mean to remove last two dimensions -> (N, 2048)
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)


class ResNetBackbone(Backbone):
  def __init__(self, architecture):
    super().__init__()

    # Backbone properties. Image preprocessing parameters are common to all
    # Torchvision ResNet models and are described in the documentation, e.g.,
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    self.feature_map_channels = 1024  # feature extractor output channels
    self.feature_pixels = 16          # ResNet feature maps are 1/16th of the original image size, similar to VGG-16 feature extractor
    self.feature_vector_size = 2048   # linear feature vector size after pooling
    self.image_preprocessing_params = image.PreprocessingParams(channel_order = image.ChannelOrder.RGB, scaling = 1.0 / 255.0, means = [ 0.485, 0.456, 0.406 ], stds = [ 0.229, 0.224, 0.225 ])

    # Construct model and pre-load with ImageNet weights
    if architecture == Architecture.ResNet101:
      resnet = ResNet_2EE(BottleNeck, [3, 4, 23, 3])
      resnet.to('cuda')
      summary(resnet, (3, 32, 32), device='cuda')
    else:
      raise ValueError("Invalid ResNet architecture value: %s" % architecture.value)
    print("Loaded IMAGENET1K_V1 pre-trained weights for Torchvision %s backbone" % architecture.value)

    # Feature extractor: given image data of shape (batch_size, channels, height, width),
    # produces a feature map of shape (batch_size, 1024, ceil(height/16), ceil(width/16))
    self.feature_extractor = FeatureExtractor(resnet = resnet)

    # Conversion of pooled features to head input
    self.pool_to_feature_vector = PoolToFeatureVector(resnet = resnet)

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
    return (self.feature_map_channels, ceil(image_height / self.feature_pixels), ceil(image_width / self.feature_pixels))