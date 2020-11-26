import numpy as np 
import time
import imutils
import cv2
import math
from  scipy.spatial import distance as dist

from skimage import util
from skimage import data, io, img_as_ubyte
import matplotlib.pyplot as plt
fgbg =cv2.bgsegm.createBackgroundSubtractorMOG()


prototext = 'mobilessd.prototxt'
model = 'mobilenetssd.caffemodel'
confThresd = 0.2

classes = ['Background', 'Aeroplane', 'Bicycle','Bird', 'Boat',
			'Bottle','Bus','Car/Auto', 'Cat','Chair',
			'Cow','Dining_table','Dog','Horse','Motorbike',
			'Person','Tree','Sheep','Sofa',
			'Train','Tvmonitor']


COLORS=np.random.uniform(0,255, size=(len(classes),3))
net = cv2.dnn.readNetFromCaffe(prototext, model)
video = 'london4.mp4'
# url ='https://192.168.43.1:8080/video'
# cam  = cv2.VideoCapture(url)
cam = cv2.VideoCapture(video)
time.sleep(2)

def func(startx,starty,endx,endy):
	xx = int((startx+endx)/2)
	yy = int((starty+endy)/2)
	box_width = abs(startx - endx)
	half_width = int(box_width/2)
	return xx,yy,half_width
boxes = []
humans  = []

width =700
height =400

while True:
	__,frame = cam.read()
	frame = imutils.resize(frame, width, height)

	h,w = frame.shape[:2]
	resize = cv2.resize(frame, (300,300))
	blob = cv2.dnn.blobFromImage(resize, 0.007843, (300,300), 127.5)
	net.setInput(blob)
	detections = net.forward()
	detShape = detections.shape[2]

	for i in np.arange(0,detShape):
		confidence = detections[0,0,i,2]
		if confidence > confThresd:
			idx = int(detections[0,0,i,1])
			if idx == 15:
				humans.append(idx)				
				box = detections[0,0,i,3:7]*np.array([w,h,w,h])
				(startx,starty,endx,endy) = box.astype('int')
				boxes.append(box.astype('int'))
				cv2.rectangle(frame, (startx,starty),(endx,endy), (0,255,0) ,2)
				label = '{} '.format(classes[idx])
				if starty-15 > 15:
					y = starty-15
				else :
					y=starty+15
				cv2.putText(frame, label, (startx,starty), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0,255,255), 1)
				if len(boxes) > 1:
					for i in range(len(boxes)):
						if i == 0:	
							x1,y1,half_width1 = func(boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3])
							cv2.circle(frame, (x1,y1), 5,  (0,255,255), -1)
							x2,y2 ,half_width2 = func(boxes[i+1][0],boxes[i+1][1],boxes[i+1][2],boxes[i+1][3])
							cv2.circle(frame, (x2,y2), 5,  (0,255,0), -1)
						else:
							x1,y1,half_width1 = func(boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3])
							cv2.circle(frame, (x1,y1), 5,  (0,255,255), -1)
							x2,y2,half_width2 = func(boxes[i-1][0],boxes[i-1][1],boxes[i-1][2],boxes[i-1][3])
							cv2.circle(frame, (x2,y2), 5,  (0,255,0), -1)

						cv2.line(frame, (x1,y1),(x2,y2), (255,0,0),2)			
						dist_F = math.sqrt(((x1-x2)**2)+((y1-y2)**2))
						dist_F = abs(dist_F - (half_width1+half_width2))
						# dist_F = dist_F*(300/(half_width1+half_width2))
						print('point distance',dist_F)
						cv2.putText(frame, '{:.1f} '.format(dist_F), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0,0,255), 2)
						if dist_F < 36:
							if i == 0:
								cv2.rectangle(frame, (boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]), (0,0,255),2)
								cv2.rectangle(frame, (boxes[i+1][0],boxes[i+1][1]),(boxes[i+1][2],boxes[i+1][3]), (0,0,255),2)
							else:
								cv2.rectangle(frame, (boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]), (0,0,255),2)
								cv2.rectangle(frame, (boxes[i-1][0],boxes[i-1][1]),(boxes[i-1][2],boxes[i-1][3]), (0,0,255),2)	


				



			
	boxes = []
	humans = []
	inverted_img = util.invert(resize)
	cv2.imshow('Object detector', frame)
	cv2.imshow('segmentation',inverted_img)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break	

cam.release()
cv2.destroyAllWindows()