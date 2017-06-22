from __future__ import print_function
import glob
import numpy as np
np.random.seed(100)
import cv2
import sys
import os
from prepare_data import get_frames_dep,get_frames_rgb,split
from prepare_model import weight_sharing


f = open(sys.argv[1]+sys.argv[2],"r").read() #sys.argv[1] is the directory and sys.argv[2] is valid_list.txt or test_list.txt
lines = f.split("\n")

if lines[-1]=='':
	lines=lines[:-1]

print("Preparing depth model")
model_dep = weight_sharing(rgb=False)
print(model_dep.summary())
model_dep.load_weights("weights-dep.h5")

print("Preparing rgb model")
model_rgb = weight_sharing(rgb=True)
print(model_rgb.summary())
model_rgb.load_weights("weights-rgb.h5")

os.chdir(sys.argv[1])

pred=[]

lines = lines[:100]
for i in range(len(lines)):
	print(lines[i])
	dep = get_frames_dep(lines[i].split(" ")[1])
	dep = split(dep,rgb=False)
	if dep.shape[0]>10:
		score_dep=np.zeros((1,249))
		for j in range(dep.shape[0]-10+1):
			score_dep += model_dep.predict(np.array([dep[j:j+10]]))
	else:
		score_dep = model_dep.predict(np.array([dep]))
	if 'train' in lines[i]:
		lines[i] = lines[i].replace("train","diff-frames")
	if 'valid' in lines[i]:
		lines[i] = lines[i].replace("valid","diff-frames-val")
	elif 'test' in sys.argv[1]:
		lines[i] = lines[i].replace("test","diff-frames-test")
	rgb = get_frames_rgb(lines[i].split(" ")[0])
	rgb = split(rgb,rgb=True)
	rgb /= 255
	if rgb.shape[0]>10:
		score_rgb=np.zeros((1,249))
		for j in range(rgb.shape[0]-10+1):
			score_rgb += model_rgb.predict(np.array([rgb[j:j+10]]))
	else:
		score_rgb = model_rgb.predict(np.array([rgb]))
	score_fin = score_dep + score_rgb
	prediction=np.argmax(score_fin,axis=1)+1
	pred.append(prediction[0])

pred = np.array(pred)

f = open(sys.argv[1]+sys.argv[2],"r").read() #sys.argv[1] is the directory and sys.argv[2] is valid_list.txt or test_list.txt
lines = f.split("\n")

if lines[-1]=='':
	lines=lines[:-1]

lines = lines[:100]

f_out = open("prediction.txt","a")

for i in range(len(lines)):
	print(lines[i]+" "+str(pred[i]),file=f_out)

f_out.close()
