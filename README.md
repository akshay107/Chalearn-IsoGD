# Chalearn-IsoGD

## Dependencies

* Python 2.7
* Keras v1.2.2 (THEANO backend)
* OpenCV-Python 3.1.0 
* matplotlib 1.5.3 or higher, numpy

##  Instructions for running the coode

* Run the file diff_frames.py to prepare the input for the RGB model. For training the model from scratch, make sure that the train folder and train_list.txt are in the same directory. Run diff_frames.py by providing the path to the file train_list.txt e.g.
```
python diff_frames.py /home/akshay/IsoGD/IsoGD_phase_1/ train_list.txt
```
A new folder named diff-frames will be created in the same directory. Similarly, you can run the program for valid_list.txt and test_list.txt which will create the folder named diff-frames-val and diff-frames-test respectively.

* For testing the model using trained weights, create the folder diff-frames-test as mentioned above. Then, run the file prepare_result.py by providing the path to the file test_list.txt e.g.
```
sudo python prepare_result.py /home/akshay/IsoGD/IsoGD_phase_2/ test_list.txt
```
**Make sure that all the test videos are inside a folder named test and test folder and test_list.txt are in the same directory.** 

This program will create a file named prediction.txt in that particular directory (in this case /home/akshay/IsoGD/IsoGD_phase_2/).

* For training the model from scratch, run the files train_rgb.py and train_dep.py by providing the path to the train folder e.g

```
sudo THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_rgb.py /home/akshay/IsoGD/IsoGD_phase_1/
sudo THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_dep.py /home/akshay/IsoGD/IsoGD_phase_1/
```
After training, model weights named weights-rgb.h5 and weights-dep.h5 will be saved in that particular directory (in this case /home/akshay/IsoGD/IsoGD_phase_1/).

Finally for testing, copy the two weight files to the same folder as prepare_result.py and then run prepare_result.py
```
sudo python prepare_result.py /home/akshay/IsoGD/IsoGD_phase_2/ test_list.txt
```
