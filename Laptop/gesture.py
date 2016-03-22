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


	def draw_hand_rect(self, frame):
		rows,cols,_ = frame.shape
		
		self.hand_row_nw = np.array([6*rows/20,6*rows/20,6*rows/20,
														10*rows/20,10*rows/20,10*rows/20,
														14*rows/20,14*rows/20,14*rows/20])

		self.hand_col_nw = np.array([9*cols/20,10*cols/20,11*cols/20,
														9*cols/20,10*cols/20,11*cols/20,
														9*cols/20,10*cols/20,11*cols/20])

		self.hand_row_se = self.hand_row_nw + 10
		self.hand_col_se = self.hand_col_nw + 10

		size = self.hand_row_nw.size
		for i in xrange(size):
			cv2.rectangle(frame,(self.hand_col_nw[i],self.hand_row_nw[i]),(self.hand_col_se[i],self.hand_row_se[i]),
										(0,255,0),1)
		#black = np.zeros(frame.shape, dtype=frame.dtype)
		#frame_final = np.vstack([black, frame])
		#return frame_final	
		return frame


	def train_hand(self, frame):
		self.set_hand_hist(frame)
		self.trained_hand = True


	def set_hand_hist(self, frame):
		#TODO use constants, only do HSV for ROI
		
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		roi = np.zeros([90,10,3], dtype=hsv.dtype)
		
		size = self.hand_row_nw.size
		for i in xrange(size):
			print frame[self.hand_row_nw[i]+20,self.hand_col_nw[i]+20]
			print hsv[self.hand_row_nw[i]+20,self.hand_col_nw[i]+20]
			print gray[self.hand_row_nw[i]+20,self.hand_col_nw[i]+20]
			roi[i*10:i*10+10,0:10] = hsv[self.hand_row_nw[i]:self.hand_row_nw[i]+10,
															 self.hand_col_nw[i]:self.hand_col_nw[i]+10]

		self.hand_hist = cv2.calcHist([roi],[0, 1], None, [180, 256], [0, 180, 0, 256])																		
		cv2.normalize(self.hand_hist, self.hand_hist, 0, 255, cv2.NORM_MINMAX)

	def apply_hist_mask(self, frame, hist): 
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv], [0,1], hist, [0,180,0,256], 1)  

		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
		cv2.filter2D(dst, -1, disc, dst)

		ret, thresh = cv2.threshold(dst, 10, 255, 0)

		kernel = np.ones((7,7),np.uint8)
		opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
		thresh = closing

		cv2.imshow('thresh',thresh)
		thresh = cv2.merge((thresh,thresh, thresh))

		res = cv2.bitwise_and(frame, thresh)
		return res


	def draw_final(self, frame):
	    decision_fb = ''
	    decision_lr = ''
	    frame_hist = self.apply_hist_mask(frame, self.hand_hist)
	    cv2.imshow('threshholded', frame_hist)
	    frame = np.zeros(frame.shape)
	    contours = self.contours(frame_hist)
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

		        #self.plot_defects(frame, defects, max_contour)

		        if centroid is not None and defects is not None and len(defects) > 0:   
		            farthest_point = self.farthest_point(defects, max_contour, centroid)

		            slope = math.atan2((farthest_point[1]-centroid[1]),(farthest_point[0]-centroid[0]))
		            incl = (slope/3.14)*180
		            print incl

		            if farthest_point is not None:
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
		if len(defects) > 0:
			for i in xrange(defects.shape[0]):
				s,e,f,d = defects[i,0]
				
				start = tuple(contour[s][0])
				end = tuple(contour[e][0])
				far = tuple(contour[f][0])
				if(self.filter_pass(start,end,far)):
					#cv2.line(frame,start,far,[0,0,255],2)
					#cv2.line(frame,end,far,[0,0,255],2)
					#cv2.circle(frame, start, 7, [255,0,255], -1)
					cv2.circle(frame, end, 7, [255,0,255], -1)
					cv2.circle(frame, far, 7, [255,255,0], -1)
    
	def filter_pass(self, start, end, far):
		angle = self.angle_lines(start, end)
		if(self.distance_points(start, far) > 10 and self.distance_points(end, far) > 10 and abs(angle) < 80):
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
	cap = cv2.VideoCapture(1)
	while(True):
		while(hd.trained_hand == False):
			ret, frame = cap.read()
			frame = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
			frame = cv2.flip(frame,1)
			frame = hd.draw_hand_rect(frame)
			cv2.imshow('frame',frame)
			k = cv2.waitKey(1)
			if k & 0xFF == ord('c'):
				hd.train_hand(frame)
				cv2.destroyAllWindows()

		while(True):
			ret, frame = cap.read()
			frame = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
			frame = cv2.flip(frame,1)
			cv2.imshow('original', frame)
			frame, decision_fb, decision_lr = hd.draw_final(frame)
			print decision_fb + ", "+decision_lr
			cv2.imshow('final',frame)
			k = cv2.waitKey(1)
			if k & 0xFF == ord('q'):	
				exit(0)	
		break
run_final()