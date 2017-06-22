import numpy as np
np.random.seed(100)
import cv2

def get_thresh(filepath):
	cap=cv2.VideoCapture(filepath)
	ret,frame=cap.read()
	frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame=np.float64(frame)/255
	pv,bl=np.histogram(frame.ravel(), bins=256, range=(0.0, 1.0))
	thresh=bl[128+np.argmax(pv[128:])]-0.1
	cap.release()
	return thresh

def get_frames_dep(filepath):
	thresh=get_thresh(filepath)
	cap=cv2.VideoCapture(filepath) #read the depth video
	ret=True
	vid=[]
	while(ret):
		ret,frame=cap.read()
		if(ret):
			frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			frame=np.float64(frame)/255
			frame[frame>thresh]=1  # to make the background white
			frame=cv2.resize(frame,(112,112))
			vid.append(frame)
	vid=np.array(vid,dtype='float32')
	vid=vid.reshape(((1,)+vid.shape))
	cap.release()
	return vid

def get_frames_rgb(filepath):
	cap=cv2.VideoCapture(filepath) #read the RGB video
	ret=True
	vid=[]
	while(ret):
		ret,frame=cap.read()
		if(ret):
			frame=cv2.resize(frame,(112,112))
			vid.append(frame)
	vid=np.array(vid,dtype='float32')
	vid=vid.transpose((3,0,1,2))
	cap.release()
	return vid

def split(frame_array,rgb):
	extra= (8-frame_array.shape[1]%8)%8# number of extra frames to add
	if(rgb):
		frame_array=np.concatenate((frame_array,np.zeros((3,extra,112,112))),axis=1)
		frame_array=frame_array.reshape((3,frame_array.shape[1]/8,8,112,112))
	else:
		frame_array=np.concatenate((frame_array,np.zeros((1,extra,112,112))),axis=1)
		frame_array=frame_array.reshape((1,frame_array.shape[1]/8,8,112,112))
	frame_array=frame_array.transpose((1,0,2,3,4))
	frame_array=frame_array.astype('float32')
	return frame_array

