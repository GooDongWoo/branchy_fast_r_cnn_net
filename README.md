# Faster R-CNN & Early Exit Net (Branchynet) in PyTorch 

## Environment Setup

Python 3.7 (for `dataclass` support) or higher is required.

Instructions here are given for Linux systems.

### PyTorch & Pip Setup

```
python -m venv pytorch_venv
source pytorch_venv/bin/activate
pip install -r pytorch/requirements.txt
```
or 
```
python -m venv pytorch_venv
source pytorch_venv/bin/activate

pip install Pillow==9.5.0
pip install pip==20.0.2
pip install setuptools==44.0.0
pip install sympy==1.12
pip install torch
pip install torchaudio
pip install torchsummary==1.5.1
pip install torchvision
pip install tqdm==4.66.2
pip install triton==2.2.0
pip install typing-extensions==4.8.0
pip install filelock==3.9.0
pip install fsspec==2023.4.0
pip install h5py==3.10.0
pip install imageio==2.34.0
pip install Jinja2==3.1.2
pip install MarkupSafe==2.1.3
pip install mpmath==1.3.0
pip install networkx==3.1
pip install numpy==1.24.1
```


## Dataset

This implementation of Faster R-CNN accepts [PASCAL Visual Object Classes](http://host.robots.ox.ac.uk/pascal/VOC/) datasets. The datasets are organized by year and VOC2007 is the default for
training and benchmarking. Images are split into `train`, `val`, and `test` splits, representing the training, validation, and test datasets. There is also a `trainval` split, which is the union of
`train` and `val`. This is what Faster R-CNN is trained on and `test` is used for validation. This is configurable on the command line.

The `download_dataset.sh` script will automatically fetch and extract VOC2007 to the default location: `VOCdevkit/VOC2007`. If your dataset is somewhere else, use `--dataset-dir` to point the program to it.



This mean? (for Linux,)
```
sh download_dataset.sh
```
or
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
echo Dataset is located in VOCdevkit/VOC2007
```
## Pre-Trained Models and Initial Weights

=> For use resnet, torchvision already have pre-trained models and initial weights. so it doesn't need to download anything.


## Running the Model

From the base directory and assuming the proper environment is configured, the PyTorch model is run like this:

```
python -m pytorch.FasterRCNN
```

Use `--help` for a summary of options or poke around the included scripts as well as `pytorch/FasterRCNN/__main__.py`. Most of the command line syntax is shared between both models. The Keras version has a few more configuration options.


### Training the Model

Numerous training parameters are available. Defaults are set to be consistent with the original paper. Some hyperparameters, such as mini-batch sampling and various detection thresholds, are hard-coded and not exposed via the command line.

Replicating the paper results requires training with stochastic gradient descent (the only option in the PyTorch version; the default in the TensorFlow version) for 10 epochs at a learning rate of 0.001 and a subsequent 4 epochs at 0.0001. The default momentum and weight decay are 0.9 and 5e-4, respectively, and image augmentation via random horizontal flips is enabled.

```
python3 -m pytorch.FasterRCNN --train --backbone=resnet101 --learning-rate=1e-3 --epochs=10  --save-best-to=results_1.pth
python3 -m pytorch.FasterRCNN --train --backbone=resnet101 --learning-rate=1e-4 --epochs=4 --load-from=results_1.pth --save-best-to=results_final.pth
```

This assumes that the dataset is present at `VOCdevkit/VOC2007/`. The mean average precision is computed from a subset of evaluation samples after each epoch and the best weights are saved at the end of training. The final model weights, regardless of accuracy, can also be saved using `--save-to` and checkpoints can be saved after each epoch to a directory using `--checkpoint-dir`.

**NOTE:** The data loader is simple but slow. If you have the CPU memory to spare (80-100 GB), `--cache-images` retains all images in memory after they are first read from disk, improving performance.

The TensorFlow version has additional options. Namely, a choice of optimizer (SGD or Adam), two RoI pooling implementations, and the option for the detector stage to output logits rather than probabilities. TensorFlow lacks an exact RoI pooling operation so by default, an approximation involving `tf.image.crop_and_resize` is used. A custom RoI pooling layer was implemented as a learning exercise but is too slow for practical use. When loading saved weights, make sure to set options consistently.

For a complete list of options use `--help`.

### Running Predictions

There are three ways to run predictions on images:

1. `--predict`: Takes a URL (local file or web URL), runs prediction, and displays the results.
2. `--predict-to-file`: Takes a URL, runs prediction, and writes the results to an image file named `predictions.png`.
3. `--predict-all`: Takes a training split from the dataset (e.g., `test`, `train`, etc.) and runs prediction on all images within it. Writes each image result to a directory named after the split (e.g., `predictions_test/`, `predictions_train/`).

Examples of each:

```
#default model is not ResNet. if you want to use ResNet model, you must not use this examples.
#instead use below examples.
python3 -m pytorch.FasterRCNN --load-from=saved_weights.pth --predict=http://trzy.org/files/fasterrcnn/gary.jpg
python3 -m pytorch.FasterRCNN --load-from=saved_weights.pth --predict-to-file=image.png
python3 -m pytorch.FasterRCNN --load-from=saved_weights.pth --predict-all=test
```

### ResNet Backbone

The PyTorch version supports different backbones. In addition to VGG-16, a few variants of ResNet (ResNet50, ResNet101, and ResNet152) are available. The `--backbone` option is used to specify one of the following backbones:

| Argument       | Backbone Description                                                                  |
|----------------|---------------------------------------------------------------------------------------|
| `vgg16`        | Custom VGG-16 backbone. The default.                                                  |
| `vgg16-torch`  | VGG-16 backbone implemented using Torchvision's pre-trained VGG-16 layers.            |
| `resnet50`     | ResNet50 backbone implemented using Torchvision's pre-trained ResNet50 layers.        |
| `resnet101`    | ResNet101 backbone implemented using Torchvision's pre-trained ResNet101 layers.      |
| `resnet152`    | ResNet152 backbone implemented using Torchvision's pre-trained ResNet152 layers.      |
| <br/>| |
| `resnet101_ee` | ResNet101 backbone with 1 additional early exit. This means resnet101_ee has 2 exits. |

All but `vgg16` load Torchvision pre-trained weights and therefore do not need to be initialized with an explicit weights file. When loading weights to resume training, the backbone must be set to be same as the one used to produce the weights. The `vgg16-torch` implementation does not accept the same weights files as `vgg16`, including `vgg16_caffe.pth`. It automatically
initializes itself using the built-in Torchvision weights (also trained on ImageNet but expecting a slightly different image pre-processing scheme) and therefore can be run without any input file. It also serves as an example of how to create a new backbone class.

Here is an example of how to train a model using a ResNet101 backbone:

```
python -m pytorch.FasterRCNN --train --backbone=resnet101 --learning-rate=1e-3 --epochs=10 --save-best-to=results_1.pth
python -m pytorch.FasterRCNN --train --backbone=resnet101 --learning-rate=1e-4 --epochs=4 --load-from=results_1.pth --save-best-to=results_final.pth
```

**Important**: When running inference with a ResNet checkpoint, the matching backbone architecture must explicitly be specified with `--backbone`, otherwise the default backbone (VGG-16) will be assumed and the checkpoint will fail to load. For example, to use the above trained ResNet101 model:

```
python -m pytorch.FasterRCNN --backbone=resnet101 --load-from=results_final.pth --predict=http://trzy.org/files/fasterrcnn/gary.jpg
```

### Saving State in PyTorch

Suppose you save your model like this:

```
t.save({
  "epoch": epoch,
  "model_state_dict": model.state_dict(),
  "optimizer_state_dict": optimizer.state_dict()
}, filename)
```

And then load it like this:

```
state = t.load(filename)
model.load_state_dict(state["model_state_dict"])
optimizer.load_state_dict(state["optimizer_state_dict"])
```

What could possibly go wrong? This left me pulling my hair out for *weeks*. My model was achieving very good mean average precision scores, comparable to the paper and other implementations, but the predicted boxes seemed ever-so-slightly worse when examined visually. The effect was subtle and random.

I went through every line of code tearing the model apart and putting it back together. At last, after glancing at these lines for what seemed like the thousandth time, it hit me: I was saving the optimizer state, as I had seen done elsewhere in PyTorch examples, but was also *loading* it again. The training process
involves changing the learning rate after 10 epochs. In my implementation, this is done by re-running the program initialized with the previous run's weights, but I was clobbering the new learning rate with the *old* one.

**Takeaway Lesson:** When saving and loading model state, pay careful attention to what it actually includes.
