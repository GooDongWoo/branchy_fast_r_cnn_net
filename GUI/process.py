import cv2
import numpy as np
import torch as t
import time
import visualize
import voc
from models import image

from models.faster_rcnn import FasterRCNNModel
from models import resnet101_ee
# from sklearn.decomposition import FastICA
if t.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available")

class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI1 = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI2 = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI3 = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.t0 = time.time()

        self.backbone = resnet101_ee.ResNetBackbone(architecture="ResNet101")
        self.model = FasterRCNNModel(
            num_classes=voc.Dataset.num_classes,
            backbone=self.backbone).cuda()
        self.load(model=self.model, filepath='./13_10_14_1.pth')

        self.EE = False

    def load(self, model, filepath):
        """
      Load model wieghts and biases from a file. We support 3 different formats:

        - PyTorch state files containing our complete model as-is
        - PyTorch state files containing only VGG-16 layers trained in Caffe (i.e.,
          the published reference implementation of VGG-16). These are compatible
          with the VGG-16 image normalization used here, unlike the torchvision
          VGG-16 implementation. The Caffe state file can be found online and is
          usually named vgg16_caffe.pth.
        - Keras h5 state file containing only VGG-16 layers trained by my own
          VGG-16 model (github.com/trzy/VGG16).

      Parameters
      ----------
      model : torch.nn.Module
        The complete Faster R-CNN model to load weights and biases into.
      filepath : str
        File to load.
      """

        state = None
        # Assume complete PyTorch state
        if state is None:
            state = t.load(filepath)
            if "model_state_dict" not in state:
                raise KeyError("Model state file '%s' is missing top-level key 'model_state_dict'" % filepath)
            state = state["model_state_dict"]

        # Load
        try:
            model.load_state_dict(state, strict=False)
            print("Loaded initial weights from '%s'" % filepath)
        except Exception as e:
            print(e)
            return

    def predict(self, model, image_data, image, show_image, output_path):
        image_data = t.from_numpy(image_data).unsqueeze(dim=0).cuda()
        scored_boxes_by_class_index = model.predict(image_data=image_data, score_threshold=0.7)
        # print("scored_boxes_by_class_index", scored_boxes_by_class_index)
        result = visualize.show_detections(
            output_path=output_path,
            show_image=show_image,
            image=image,
            scored_boxes_by_class_index=scored_boxes_by_class_index,
            class_index_to_name=voc.Dataset.class_index_to_name)
        return result

    def predict_ee(self, model, image_data, image, show_image, output_path):
        image_data = t.from_numpy(image_data).unsqueeze(dim=0).cuda()
        scored_boxes_by_class_index = model.predict_ee(image_data=image_data, score_threshold=0.7)
        # print("scored_boxes_by_class_index_ee", scored_boxes_by_class_index)
        result = visualize.show_detections(
            output_path=output_path,
            show_image=show_image,
            image=image,
            scored_boxes_by_class_index=scored_boxes_by_class_index,
            class_index_to_name=voc.Dataset.class_index_to_name)
        return result

    def run(self):
        # frame, face_frame, ROI1, ROI2, status, mask = self.fd.face_detect(self.frame_in)

        # EE 출력 결정에 관한 알고리즘 작성.
        # EE 출력으로 결정 된다면 self.EE = True
        # Last Layer 출력으로 결정 된다면 self.EE = False



        frame = self.frame_in
        # print(frame.shape)
        # AA = "000002.png"
        image_data, image_obj, _, _ = image.load_image(url=frame,
                                                       preprocessing=self.model.backbone.image_preprocessing_params, min_dimension_pixels = 600)
        result1 = self.predict_ee(model=self.model, image_data=image_data, image=image_obj, show_image=False,
                                 output_path=None)
        result1 = np.array(result1)
        self.frame_ROI1 = result1

        image_data, image_obj, _, _ = image.load_image(url=frame,
                                                       preprocessing=self.model.backbone.image_preprocessing_params,
                                                       min_dimension_pixels=600)

        result2 = self.predict(model=self.model, image_data=image_data, image=image_obj, show_image=False,
                              output_path=None)
        result2 = np.array(result2)
        self.frame_out = frame
        self.frame_ROI2 = result2
        self.frame_ROI3 = result2
        return True
    
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI1 = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI2 = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.t0 = time.time()
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
