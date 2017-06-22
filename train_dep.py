from __future__ import print_function
import glob
import numpy as np
np.random.seed(100)
import cv2
import sys
import os
from prepare_data import get_frames_dep,split
from prepare_model import weight_sharing
from keras.callbacks import LearningRateScheduler

def generate_data(lines):
	batches = [lines[x:x+200] for x in range(0, len(lines), 200)]
	while True:
		for b in range(len(batches)):
			l1=[]
			l2=[]
			for counter in range(0,len(batches[b]),2):
				data=[]
				label=[]
				for line in batches[b][counter:counter+2]:
					dep = get_frames_dep(line.split(" ")[1])
					dep = split(dep,rgb=False)
					data.append(dep)
					label.append(int(line.split(" ")[2])-1) # -1 because the output_label should be between 0 and nb_classes-1
				X_train=np.array(data)
				Y_train=np.array(label)
				(new4,new5)=zip(*sorted(zip(X_train,Y_train),key=lambda x:x[0].shape[0]))
				new4=list(new4)
				new5=list(new5)
				d={}
				for i in range(len(new4)):
					if(new4[i].shape[0] not in d.keys()):
						d[new4[i].shape[0]]=1
					else:
						d[new4[i].shape[0]]+=1
				l=[d[key] for key in sorted(d.keys())]
				l=np.cumsum(l)
				for i in range(len(l)):
					if i==0:
						l1.append(np.array(new4[:l[i]]))
						l2.append(np.array(new5[:l[i]]))
					else:
						l1.append(np.array(new4[l[i-1]:l[i]]))
						l2.append(np.array(new5[l[i-1]:l[i]]))
			for i in range(len(l1)):
				if l1[i].shape[1]>10:
					ind=range(l1[i].shape[1]-10+1)
					ind=np.random.permutation(ind)[0]
					l1[i]=l1[i][:,ind:ind+10,:,:,:,:]
				x=l1[i]
				y=l2[i]
				yield (x,y)

def step_decay(epoch):
	if epoch<=4:
		lrate = 0.005
	elif epoch>4 and epoch<=9:
		lrate = 0.0025
	elif epoch>9:
		lrate = 0.00125
	return lrate

'''
sys.argv[1] is the directory of train_lines_new.txt and val_lines_new.txt. Make sure it's the same directory where train folder is located
'''

f_train=open(sys.argv[1]+"train_lines_new.txt","r").read() 
train_lines=f_train.split("\n")
train_lines=train_lines[:len(train_lines)-1]
f_val=open(sys.argv[1]+"val_lines_new.txt","r").read()
val_lines=f_val.split("\n")
val_lines=val_lines[:len(val_lines)-1]
print(len(train_lines),len(val_lines))
print("Preparing model")
model=weight_sharing(rgb=False)
print(model.summary())
print("Model ready. Starting training")
os.chdir(sys.argv[1])
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
model.fit_generator(generate_data(train_lines),samples_per_epoch=len(train_lines),nb_epoch=15,callbacks=callbacks_list)
score=model.evaluate_generator(generate_data(val_lines),val_samples=len(val_lines))
print(score)
model.save_weights("weights-dep.h5",overwrite=True)
print("Weights saved to file weights-dep.h5 in the same directory where train folder is located")
