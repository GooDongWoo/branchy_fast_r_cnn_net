# GUI for Real - Time Image Dection with Faster R-CNN & Early Exit Net(Branchynet) in PYQT 

## Environment Setup

Python 3.7 (for `dataclass` support) or higher is required.

Instructions here are given for Linux systems.

### PyTorch & Pip Setup

Similar as " Faster R-CNN & Early Exit Net(Branchynet) in PyTorch"

In " Faster R-CNN & Early Exit Net(Branchynet) in PyTorch" Pip Setting,
Add unders.
```
python -m venv pytorch_venv
source pytorch_venv/bin/activate

pip install opencv-python
pip install pyqt5

```


## Trained weight

When you already prepare trained weight file as '....pth,'
you need to setting not only '....pth' file, but also 'InterFE.pth', 'layer1.pth', ... 'layer4.pth' which is pretrained weight for Early Exit Resnet101 model. ( we use these pretrained Early Exit Resnet101 weight to build Faster R-CNN models. )

Put these files just on './GUI'

