Fine Tuning Example of Chainer using VGG

# Preparation

```
wget http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
python convert_caffemodel_to_npz.py
```

# Usage

Train all layers (not fine tuning)

```
python train.py
```

Train fc8, fc7, fc6 layers, Freeze other layers

```
python train.py -m <Pretrained Model Npz File>
```
