# Chalearn-IsoGD

## Dependencies

The code was written in python 2.7 and uses following packages
* Theano 0.8.2 (https://github.com/Theano/Theano/tree/0.8.X)
* Keras v1.2.2 (THEANO backend) (https://github.com/fchollet/keras/tree/keras-1)
* OpenCV-Python 3.1.0 with CUDA support (To install, follow these instructions: http://www.pyimagesearch.com/2016/07/11/compiling-opencv-with-cuda-support/) 
* matplotlib 1.5.3 or higher, numpy

##  Instructions for training the model

* **Make sure that the train folder and train_list.txt are in the same directory.** Run diff_frames.py by providing the path to the file train_list.txt e.g.
```
python diff_frames.py /home/akshay/IsoGD/IsoGD_phase_1/ train_list.txt
```
A new folder named diff-frames will be created in the directory provided (in this case /home/akshay/IsoGD/IsoGD_phase_1/).

* Run the files train_rgb.py and train_dep.py by providing the path to the train folder and setting theano environment variable e.g

```
sudo THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_rgb.py /home/akshay/IsoGD/IsoGD_phase_1/
sudo THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_dep.py /home/akshay/IsoGD/IsoGD_phase_1/
```
After training, model weights will be saved in the file weights-rgb.h5 and weights-dep.h5  in that particular directory
(in this case /home/akshay/IsoGD/IsoGD_phase_1/).

## Instructions for testing the model

* For testing the model, **make sure that all the test videos are inside a folder named test e.g. the first test video 
is /home/akshay/IsoGD/IsoGD_phase_2/test/001/M_00001.avi and test folder and  the file test_list.txt are in the same directory.** Run diff_frames.py by providing the path to the file test_list.txt e.g.

```
python diff_frames.py /home/akshay/IsoGD/IsoGD_phase_2/ test_list.txt
```
A new folder named diff-frames-test will be created in the directory provided (in this case /home/akshay/IsoGD/IsoGD_phase_2/).

* Copy the two weight files (weights-rgb.h5 and weights-dep.h5) to the same folder where this repository was downloaded. 
You can either use the weight files in this repository or the weight files obtained after you trained the model.

* Run the file prepare_result.py by providing path to the file test_list.txt e.g.
```
sudo python prepare_result.py /home/akshay/IsoGD/IsoGD_phase_2/ test_list.txt
```
This program will create a file named prediction.txt in that particular directory (in this case /home/akshay/IsoGD/IsoGD_phase_2/).
