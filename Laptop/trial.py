from gesture import HandDetection
import socket

hd = HandDetection()
cap = cv2.VideoCapture(0)

def initialize():
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

def run_once():	
	ret, frame = cap.read()
	frame = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
	frame = cv2.flip(frame,1)
	cv2.imshow('original', frame)
	frame, decision_fb, decision_lr = hd.draw_final(frame)
	cv2.imshow('final',frame)
	k = cv2.waitKey(1)
	if k & 0xFF == ord('q'):	
		exit(0)	
	return [decision_fb, decision_lr]


serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serverSocket.bind((socket.gethostname(),10000))
serverSocket.listen(1)
initialize()

while(True):
	clientSocket, addr = serverSocket.accept()
	print "Car connected..."
	while(True):
		decision_fb, decision_lr = run_once()
		decision = 0
		if decision_fb == 'f' and decision_lr == '':
			decision = 0
		elif decision_fb == 'f' and decision_lr == 'l':
			decision = 2
		elif decision_fb == 'f' and decision_lr == 'r':
			decision = 3
		elif decision_fb == 'b':
			decision = 1
		else:
			pass

		clientSocket.send(decision)



