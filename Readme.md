# EDRAM-Tensorflow
[![Packagist](https://img.shields.io/badge/Tensorflow-1.3.0-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Tensorlayer-1.6.1-blue.svg)]()

Abstract
---
This is the re-implementation of the original paper - Enriched Deep Recurrent Visual Attention Model[1]. Moreover, it's the improvement toward the deepmind original paper([repo](https://github.com/SunnerLi/ram)). In September 2017, there's only one [implementation](https://github.com/ablavatski/EDRAM) which is written in theano. As the result, I tries to re-write as the tensorflow version. To be notice, **this is not the official repository**.    

<br/>

Model Structure
---

![](https://github.com/SunnerLi/EDRAM-Tensorflow/blob/master/img/structure.jpg)

Issue
---
I following the whole process which are written in theano. However, the slow converge phenomenon it shows. I think there're 2 possible reasons.    
1. I might use LSTM API incorrectly, so the model didn't use the temporal information to get more accurate location result. 
2. In the original implementation, the authors define spacial batch normalization and mlp batch normalization by themselves. However, to simplify the work, I just use `SpatialTransformer2dAffineLayer` to perform spacial transformation mechanism which is provided by tensorlayer. Maybe this change influence the result.    

<br/>

How to Get Training Data
---
This paragraph arranges the order step how to collect the training data.
To be notice, you should install the **lua** interpreter and **torch** before generating the data.
1. install luajit
```
$ sudo apt-get install luajit
```

2. install torch by the official web steps
(This step might cost some time...)
```
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; bash install-deps;
$ ./install.sh
```

3. Add torch environment PATH
```
$ export LD_LIBRARY_PATH=/home/sunner/torch/install/bin/torch-activate
$ source ~/.bashrc
```

4. clone the DeepMind repository and download the data
```
$ git clone https://github.com/ablavatski/mnist-cluttered.git
$ cd mnist-cluttered
$ luajit download_mnist.lua
```

5. generate the data
```
$ luajit mnist_cluttered_gen.lua
$ python mnist_cluttered.py --path mnist_clusttered.hdf5
```

The `mnist_clusttered.hdf5` is the final dataset file.    

<br/>

Reference
---
[1]	A.Ablavatski, S.Lu, and J.Cai, “Enriched Deep Recurrent Visual Attention Model for Multiple Object Recognition,” _Arxiv_, 2017.