import numpy as np
import cv2
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import sys

f=open(sys.argv[1]+sys.argv[2],"r").read()  #sys.argv[1] is the directory and sys.argv[2] is train_list.txt,valid_list.txt or test_list.txt
lines=f.split("\n")

if lines[-1]=='':
	lines=lines[:-1]

os.chdir(sys.argv[1])
for t in range(len(lines)):
	line=lines[t].split(' ')[0]
	print(line)
	frames=[]
	cap=cv2.VideoCapture(line)
	ret=True
	while(ret):
		ret,frame=cap.read()
		frames.append(frame)
	frames=frames[:-1]
	diff_frames=[]
	for l in range(len(frames)):
		x=np.float64(frames[l])
		y=np.float64(frames[0])
		I=np.abs(x-y)
		I=np.sum(I,axis=2)
		diff_frames.append(I)
	if 'train' in sys.argv[2]:
		out_line=line.replace("train","diff-frames")
	elif 'valid' in sys.argv[2]:
		out_line=line.replace("valid","diff-frames-val")
	elif 'test' in sys.argv[2]:
		out_line=line.replace("test","diff-frames-test")
	out_dir=re.sub(r"M_\d+.avi*",r"",out_line)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	cm_jet = mpl.cm.get_cmap('jet')
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(out_line,fourcc, cap.get(cv2.CAP_PROP_FPS), (frames[l].shape[1],frames[l].shape[0]))
	for i in range(1,len(diff_frames)):
		dz=(diff_frames[i]-np.min(diff_frames[i]))/(np.max(diff_frames[i])-np.min(diff_frames[i]))
		im=cm_jet(dz)
		im=im[:,:,:3]
		im*=255
		im=np.uint8(im)
		im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
		out.write(im)
	out.release()
	cap.release()
