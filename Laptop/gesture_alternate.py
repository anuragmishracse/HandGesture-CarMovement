import cv2
import numpy as np
import pyautogui
import math


class HandDetection:
	def __init__(self):		
		self.trained_hand = False
		self.hand_row_nw = None
		self.hand_row_se = None
		self.hand_col_nw = None
		self.hand_col_se = None
		self.hand_hist = None
		self.previous_points =[]


	def analyse(self, frame):
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(5,5),0)
		ret,thresh = cv2.threshold(blur, 50 ,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

		cv2.imshow('thresh',thresh)
		thresh = cv2.merge((thresh,thresh, thresh))

		res = cv2.bitwise_and(frame, thresh)
		return res

		return thresh


	def draw_final(self, frame):
		decision_fb = ''
		decision_lr = ''
		roi = frame[0:frame.shape[0]/2, int(frame.shape[1]*2/3.0):frame.shape[1]]
		cv2.imshow('roi', roi)
		roi_thresholded = self.analyse(roi)
		cv2.imshow('roi thresholded', roi_thresholded)

		#frame = np.zeros(roi.shape)
		frame = roi
		contours = self.contours(roi_thresholded)
		if contours is not None and len(contours) > 0:
			max_contour = self.max_contour(contours)
			if max_contour is not None:
				self.plot_contours(frame, max_contour)
				#x,y,w,h = cv2.boundingRect(max_contour)
				#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),0)
				hull = self.hull(max_contour)
				self.plot_hull(frame, hull)
				centroid = self.centroid(max_contour)
				self.plot_centroid(frame, centroid)
				self.plot_previous(frame, centroid)
				defects = self.defects(max_contour)

				valid_defects = self.plot_defects(frame, defects, max_contour)

				if centroid is not None and defects is not None and len(defects) > 0 and valid_defects < 3 and valid_defects > 0:   
					farthest_point = self.farthest_point(defects, max_contour, centroid)

					if farthest_point is not None:

						slope = math.atan2((farthest_point[1]-centroid[1]),(farthest_point[0]-centroid[0]))
						incl = (slope/3.14)*180


						self.plot_farthest_point(frame, farthest_point)
						if (incl > -45.0 and incl < 45.0) or (incl < -135.0 or incl > 135.0):
						    if farthest_point[0] <= centroid[0]:
						    	decision_lr = 'l'
						    else:
						    	decision_lr = 'r'

						if farthest_point[1] <= centroid[1]:
							decision_fb = 'f'
						else:
							decision_fb = 'b'


		                #size = pyautogui.size()
		                #x_inc = size[0]/float(frame.shape[1])
		                #y_inc = size[1]/float(frame.shape[0])
		                #position = pyautogui.position()	
		                # pyautogui.moveTo(int(farthest_point[0]*x_inc), int(farthest_point[1]*y_inc), duration=0)
		                #print "Point : "+str(farthest_point)+", Cursor : "+str(farthest_point[0]*x_inc)+"-"+str(farthest_point[1]*y_inc)
		return [frame, decision_fb, decision_lr]

	def plot_previous(self, frame, new_point):
		n=10
		for point in self.previous_points:
			cv2.circle(frame, point, 3, [255,255,255], -1)

		if(len(self.previous_points)!=n):
			self.previous_points.append(new_point)
		else:
			for i in xrange(n-1):
				self.previous_points[i] = self.previous_points[i+1]
			self.previous_points[n-1] = new_point

	def plot_defects(self, frame, defects, contour):
		counter = 0
		length = self.distance_points((int(frame.shape[1]*2/3.0), 0), (int(frame.shape[1]*2/3.0), frame.shape[0]/2))
		if defects != None:
			if len(defects) > 0:
				for i in xrange(defects.shape[0]):
					s,e,f,d = defects[i,0]
					
					start = tuple(contour[s][0])
					end = tuple(contour[e][0])
					far = tuple(contour[f][0])
					if(self.filter_pass(start,end,far, length)):
						#cv2.line(frame,start,far,[0,0,255],2)
						#cv2.line(frame,end,far,[0,0,255],2)
						#cv2.circle(frame, start, 7, [255,0,255], -1)
						cv2.circle(frame, end, 7, [255,0,255], -1)
						cv2.circle(frame, far, 7, [255,255,0], -1)
						counter+=1
		return counter
	    
	def filter_pass(self, start, end, far, length):
		angle = self.angle_lines(start, end)
		if(self.distance_points(start, far) > length/4.0 and self.distance_points(end, far) > length/4.0 and abs(angle) < 80):
			return 1
		else:
			return 0

	def distance_points(self, start, end):
		return cv2.sqrt(cv2.add(cv2.pow(int(start[0]-end[0]),2),cv2.pow(int(start[1]-end[1]),2)))[0][0]

	def angle_lines(self, start, end):
		angle = int(math.atan2((start[1]-end[1]),(start[0]-end[0]))*180/math.pi)
		return angle

	def plot_farthest_point(self, frame, point):
		cv2.circle(frame, point, 5, [0,0,255], -1)

	def plot_centroid(self, frame, point):
		cv2.circle(frame, point, 5, [255,0,0], -1)

	
	def plot_hull(self, frame, hull):
		cv2.drawContours(frame, [hull], 0, (255,0,0), 2)	


	def plot_contours(self, frame, contours):
		cv2.drawContours(frame, contours, -1, (0,255,0), 2)


	def contours(self,frame):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(gray, 0, 255, 0)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)	
		#contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		return contours

	def max_contour(self,contours):
		max_i = 0
		max_area = 0
		
		for i in range(len(contours)):
			cnt = contours[i]
			area = cv2.contourArea(cnt)
			if area > max_area:
				max_area = area
				max_i = i

		contour = contours[max_i]
		return contour

	def hull(self,contour):
		hull = cv2.convexHull(contour)
		return hull

	def defects(self,contour):
		hull = cv2.convexHull(contour, returnPoints=False)
		if hull is not None and len(hull > 3) and len(contour) > 3:
			defects = cv2.convexityDefects(contour, hull)	
			return defects
		else: 
			return None

	def centroid(self,contour):
		moments = cv2.moments(contour)
		if moments['m00'] != 0:
			cx = int(moments['m10']/moments['m00'])
			cy = int(moments['m01']/moments['m00'])
			return (cx,cy)
		else:
			return None		


	def farthest_point(self, defects, contour, centroid):
		s = defects[:,0][:,0]
		cx, cy = centroid
		
		x = np.array(contour[s][:,0][:,0], dtype=np.float)
		y = np.array(contour[s][:,0][:,1], dtype=np.float)
					
		xp = cv2.pow(cv2.subtract(x, cx), 2)
		yp = cv2.pow(cv2.subtract(y, cy), 2)
		dist = cv2.sqrt(cv2.add(xp, yp))

		dist_max_i = np.argmax(dist)

		if dist_max_i < len(s):
			farthest_defect = s[dist_max_i]
			farthest_point = tuple(contour[farthest_defect][0])
			return farthest_point
		else:
			return None


def run_final():

	hd = HandDetection()
	cap = cv2.VideoCapture(0)

	while(True):
		ret, frame = cap.read()
		#frame = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
		frame = cv2.flip(frame,1)
		cv2.rectangle(frame, (frame.shape[1],0),(int(frame.shape[1]*2/3.0), frame.shape[0]/2),(0,255,0),2 )
		cv2.imshow('original', frame)
		frame, decision_fb, decision_lr = hd.draw_final(frame)
		print decision_fb + ", "+decision_lr
		cv2.imshow('final',frame)
		k = cv2.waitKey(1)
		if k & 0xFF == ord('q'):	
			exit(0)	
#run_final()
