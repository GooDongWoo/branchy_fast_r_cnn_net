#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# pytorch/FasterRCNN/models/faster_rcnn.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# PyTorch implementation of Faster R-CNN training and inference models. Here,
# all stages of Faster R-CNN are instantiated, RPN mini-batches are sampled,
# ground truth labels from RPN proposal boxes (RoIs) for the detector stage are
# generated, and  proposals are sampled.
#

from dataclasses import dataclass
import numpy as np
import random
import torch as t
from torch import nn
from torchvision.ops import nms

from . import utils
from . import anchors
from . import math_utils
from . import rpn
from . import detector


class FasterRCNNModel(nn.Module):
    @dataclass
    class Loss:
        rpn_class: float
        rpn_regression: float
        detector_class: float
        detector_regression: float
        total: float

    def __init__(self, num_classes, backbone, rpn_minibatch_size=256, proposal_batch_size=128,
                 allow_edge_proposals=True):
        """
    Parameters
    ----------
    num_classes : int
      Number of output classes.
    backbone : models.Backbone
      Backbone network for feature extraction and pooled feature vector
      construction (for input to detector heads).
    rpn_minibatch_size : int
      Size of the RPN mini-batch. The number of ground truth anchors sampled
      for training at each step.
    proposal_batch_size : int
      Number of region proposals to sample at each training step.
    allow_edge_proposals : bool
      Whether to use proposals generated at invalid anchors (those that
      straddle image edges). Invalid anchors are excluded from RPN training, as
      explicitly stated in the literature, but Faster R-CNN implementations
      tend to still pass proposals generated at invalid anchors to the
      detector.
    """
        super().__init__()

        # Constants
        self._num_classes = num_classes
        self._rpn_minibatch_size = rpn_minibatch_size
        self._proposal_batch_size = proposal_batch_size
        self._detector_box_delta_means = [0, 0, 0, 0]
        self._detector_box_delta_stds = [0.1, 0.1, 0.2, 0.2]

        # Backbone
        self.backbone = backbone

        # Network stages
        self._stage1_feature_extractor = backbone.feature_extractor
        self._stage2_region_proposal_network = rpn.RegionProposalNetwork(
            feature_map_channels=backbone.feature_map_channels,
            allow_edge_proposals=allow_edge_proposals
        )
        self._stage3_detector_network = detector.DetectorNetwork(
            num_classes=num_classes,
            backbone=backbone
        )

    def forward(self, image_data, anchor_map=None, anchor_valid_map=None):
        """
    Forward inference. Use for test and evaluation only.

    Parameters
    ----------
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized using the VGG-16 convention (BGR, ImageNet channel-wise
      mean-centered).
    anchor_map : torch.Tensor
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map. If this or anchor_valid_map is not
      provided, both will be computed here.
    anchor_valid_map : torch.Tensor
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width). If this or anchor_map is not provided, both will
      be computed here.

    Returns
    -------
    np.ndarray, torch.Tensor, torch.Tensor
      - Proposals (N, 4) from region proposal network
      - Classes (M, num_classes) from detector network
      - Box delta regressions (M, (num_classes - 1) * 4) from detector network
    """
        assert image_data.shape[0] == 1, "Batch size must be 1"
        image_shape = image_data.shape[1:]  # (batch_index, channels, height, width) -> (channels, height, width)

        # Anchor maps can be pre-computed and passed in explicitly (for performance
        # reasons) but if they are missing, we compute them on-the-fly here
        if anchor_map is None or anchor_valid_map is None:
            feature_map_shape = self.backbone.compute_feature_map_shape(image_shape=image_shape)
            anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape=image_shape,
                                                                        feature_map_shape=feature_map_shape,
                                                                        feature_pixels=self.backbone.feature_pixels)

        # Run each stage
        feature_map = self._stage1_feature_extractor(image_data=image_data)
        # print("anchor_map", anchor_map.shape)
        # print("anchor_valid_map", anchor_valid_map.shape)
        objectness_score_map, box_deltas_map, proposals = self._stage2_region_proposal_network(
            feature_map=feature_map,
            image_shape=image_shape,
            anchor_map=anchor_map,
            anchor_valid_map=anchor_valid_map,
            max_proposals_pre_nms=6000,  # test time values
            max_proposals_post_nms=300
        )
        classes, box_deltas = self._stage3_detector_network(
            feature_map=feature_map,
            proposals=proposals
        )

        return proposals, classes, box_deltas

    def earlyexit(self, image_data, anchor_map=None, anchor_valid_map=None):
        """
    Forward inference. Use for test and evaluation only.

    Parameters
    ----------
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized using the VGG-16 convention (BGR, ImageNet channel-wise
      mean-centered).
    anchor_map : torch.Tensor
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map. If this or anchor_valid_map is not
      provided, both will be computed here.
    anchor_valid_map : torch.Tensor
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width). If this or anchor_map is not provided, both will
      be computed here.

    Returns
    -------
    np.ndarray, torch.Tensor, torch.Tensor
      - Proposals (N, 4) from region proposal network
      - Classes (M, num_classes) from detector network
      - Box delta regressions (M, (num_classes - 1) * 4) from detector network
    """
        assert image_data.shape[0] == 1, "Batch size must be 1"
        image_shape = image_data.shape[1:]  # (batch_index, channels, height, width) -> (channels, height, width)

        # Anchor maps can be pre-computed and passed in explicitly (for performance
        # reasons) but if they are missing, we compute them on-the-fly here
        if anchor_map is None or anchor_valid_map is None:
            feature_map_shape = self.backbone.compute_feature_map_shape(image_shape=image_shape)
            anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape=image_shape,
                                                                        feature_map_shape=feature_map_shape,
                                                                        feature_pixels=self.backbone.feature_pixels)

        # Run each stage
        feature_map_a = self._stage1_feature_extractor.forward_ee(image_data=image_data)
        objectness_score_map, box_deltas_map, proposals = self._stage2_region_proposal_network(
            feature_map=feature_map_a,
            image_shape=image_shape,
            anchor_map=anchor_map,
            anchor_valid_map=anchor_valid_map,
            max_proposals_pre_nms=6000,  # test time values
            max_proposals_post_nms=300
        )
        classes, box_deltas = self._stage3_detector_network(
            feature_map=feature_map_a,
            proposals=proposals
        )

        return proposals, classes, box_deltas

    @utils.no_grad
    def predict(self, image_data, score_threshold, anchor_map=None, anchor_valid_map=None):
        """
    Performs inference on an image and obtains the final detected boxes.

    Parameters
    ----------
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized using the VGG-16 convention (BGR, ImageNet channel-wise
      mean-centered).
    score_threshold : float
      Minimum required score threshold (applied per class) for a detection to
      be considered. Set this higher for visualization to minimize extraneous
      boxes.
    anchor_map : torch.Tensor
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map. If this or anchor_valid_map is not
      provided, both will be computed here.
    anchor_valid_map : torch.Tensor
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width). If this or anchor_map is not provided, both will
      be computed here.

    Returns
    -------
    Dict[int, np.ndarray]
      Scored boxes, (N, 5) tensor of box corners and class score,
      (y1, x1, y2, x2, score), indexed by class index.
    """
        self.eval()
        assert image_data.shape[0] == 1, "Batch size must be 1"

        # Forward inference
        proposals, classes, box_deltas = self(
            image_data=image_data,
            anchor_map=anchor_map,
            anchor_valid_map=anchor_valid_map
        )
        proposals = proposals.cpu().numpy()
        classes = classes.cpu().numpy()
        box_deltas = box_deltas.cpu().numpy()

        # Convert proposal boxes -> center point and size
        proposal_anchors = np.empty(proposals.shape)
        proposal_anchors[:, 0] = 0.5 * (proposals[:, 0] + proposals[:, 2])  # center_y
        proposal_anchors[:, 1] = 0.5 * (proposals[:, 1] + proposals[:, 3])  # center_x
        proposal_anchors[:, 2:4] = proposals[:, 2:4] - proposals[:, 0:2]  # height, width

        # Separate out results per class: class_idx -> (y1, x1, y2, x2, score)
        boxes_and_scores_by_class_idx = {}
        for class_idx in range(1, classes.shape[1]):  # skip class 0 (background)
            # Get the box deltas (ty, tx, th, tw) corresponding to this class, for
            # all proposals
            box_delta_idx = (class_idx - 1) * 4
            box_delta_params = box_deltas[:, (box_delta_idx + 0): (box_delta_idx + 4)]  # (N, 4)
            proposal_boxes_this_class = math_utils.convert_deltas_to_boxes(
                box_deltas=box_delta_params,
                anchors=proposal_anchors,
                box_delta_means=self._detector_box_delta_means,
                box_delta_stds=self._detector_box_delta_stds
            )

            # Clip to image boundaries
            proposal_boxes_this_class[:, 0::2] = np.clip(proposal_boxes_this_class[:, 0::2], 0,
                                                         image_data.shape[2] - 1)  # clip y1 and y2 to [0,height)
            proposal_boxes_this_class[:, 1::2] = np.clip(proposal_boxes_this_class[:, 1::2], 0,
                                                         image_data.shape[3] - 1)  # clip x1 and x2 to [0,width)

            # Get the scores for this class. The class scores are returned in
            # normalized categorical form. Each row corresponds to a class.
            scores_this_class = classes[:, class_idx]

            # Keep only those scoring high enough
            sufficiently_scoring_idxs = np.where(scores_this_class > score_threshold)[0]
            proposal_boxes_this_class = proposal_boxes_this_class[sufficiently_scoring_idxs]
            scores_this_class = scores_this_class[sufficiently_scoring_idxs]
            boxes_and_scores_by_class_idx[class_idx] = (proposal_boxes_this_class, scores_this_class)

        # Perform NMS per class
        scored_boxes_by_class_idx = {}
        # print("boxes_and_scores_by_class_idx", boxes_and_scores_by_class_idx)
        for class_idx, (boxes, scores) in boxes_and_scores_by_class_idx.items():
            idxs = nms(
                boxes=t.from_numpy(boxes).cuda(),
                scores=t.from_numpy(scores).cuda(),
                iou_threshold=0.3
                # TODO: unsure about this. Paper seems to imply 0.5 but https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py has 0.3 for test NMS
            ).cpu().numpy()
            boxes = boxes[idxs]
            scores = np.expand_dims(scores[idxs], axis=0)  # (N,) -> (N,1)
            scored_boxes = np.hstack([boxes, scores.T])  # (N,5), with each row: (y1, x1, y2, x2, score)
            scored_boxes_by_class_idx[class_idx] = scored_boxes

        return scored_boxes_by_class_idx

    def predict_ee(self, image_data, score_threshold, anchor_map=None, anchor_valid_map=None):
        """
    Performs inference on an image and obtains the final detected boxes.

    Parameters
    ----------
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized using the VGG-16 convention (BGR, ImageNet channel-wise
      mean-centered).
    score_threshold : float
      Minimum required score threshold (applied per class) for a detection to
      be considered. Set this higher for visualization to minimize extraneous
      boxes.
    anchor_map : torch.Tensor
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map. If this or anchor_valid_map is not
      provided, both will be computed here.
    anchor_valid_map : torch.Tensor
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width). If this or anchor_map is not provided, both will
      be computed here.

    Returns
    -------
    Dict[int, np.ndarray]
      Scored boxes, (N, 5) tensor of box corners and class score,
      (y1, x1, y2, x2, score), indexed by class index.
    """
        self.eval()
        assert image_data.shape[0] == 1, "Batch size must be 1"

        # Forward inference
        proposals, classes, box_deltas = self.earlyexit(
            image_data=image_data,
            anchor_map=anchor_map,
            anchor_valid_map=anchor_valid_map
        )

        proposals = proposals.cpu().numpy()
        classes = classes.cpu().detach().numpy()
        box_deltas = box_deltas.cpu().detach().numpy()

        # Convert proposal boxes -> center point and size
        proposal_anchors = np.empty(proposals.shape)
        proposal_anchors[:, 0] = 0.5 * (proposals[:, 0] + proposals[:, 2])  # center_y
        proposal_anchors[:, 1] = 0.5 * (proposals[:, 1] + proposals[:, 3])  # center_x
        proposal_anchors[:, 2:4] = proposals[:, 2:4] - proposals[:, 0:2]  # height, width

        # Separate out results per class: class_idx -> (y1, x1, y2, x2, score)
        boxes_and_scores_by_class_idx = {}
        for class_idx in range(1, classes.shape[1]):  # skip class 0 (background)
            # Get the box deltas (ty, tx, th, tw) corresponding to this class, for
            # all proposals
            box_delta_idx = (class_idx - 1) * 4
            box_delta_params = box_deltas[:, (box_delta_idx + 0): (box_delta_idx + 4)]  # (N, 4)
            proposal_boxes_this_class = math_utils.convert_deltas_to_boxes(
                box_deltas=box_delta_params,
                anchors=proposal_anchors,
                box_delta_means=self._detector_box_delta_means,
                box_delta_stds=self._detector_box_delta_stds
            )

            # Clip to image boundaries
            proposal_boxes_this_class[:, 0::2] = np.clip(proposal_boxes_this_class[:, 0::2], 0,
                                                         image_data.shape[2] - 1)  # clip y1 and y2 to [0,height)
            proposal_boxes_this_class[:, 1::2] = np.clip(proposal_boxes_this_class[:, 1::2], 0,
                                                         image_data.shape[3] - 1)  # clip x1 and x2 to [0,width)

            # Get the scores for this class. The class scores are returned in
            # normalized categorical form. Each row corresponds to a class.
            scores_this_class = classes[:, class_idx]

            # Keep only those scoring high enough
            sufficiently_scoring_idxs = np.where(scores_this_class > score_threshold)[0]
            proposal_boxes_this_class = proposal_boxes_this_class[sufficiently_scoring_idxs]
            scores_this_class = scores_this_class[sufficiently_scoring_idxs]
            boxes_and_scores_by_class_idx[class_idx] = (proposal_boxes_this_class, scores_this_class)

        # Perform NMS per class
        scored_boxes_by_class_idx = {}
        for class_idx, (boxes, scores) in boxes_and_scores_by_class_idx.items():
            idxs = nms(
                boxes=t.from_numpy(boxes).cuda(),
                scores=t.from_numpy(scores).cuda(),
                iou_threshold=0.3
                # TODO: unsure about this. Paper seems to imply 0.5 but https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py has 0.3 for test NMS
            ).cpu().numpy()
            boxes = boxes[idxs]
            scores = np.expand_dims(scores[idxs], axis=0)  # (N,) -> (N,1)
            scored_boxes = np.hstack([boxes, scores.T])  # (N,5), with each row: (y1, x1, y2, x2, score)
            scored_boxes_by_class_idx[class_idx] = scored_boxes

        return scored_boxes_by_class_idx
