import cv2
<<<<<<< HEAD
import time
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()
=======
import numpy as np

#ShiTomasi corner detection
feature_params = dict(maxCorners=200,  qualityLevel=0.01, minDistance=30, blockSize=3)

#Lucas-Kanade optical flow,Gunnar Farneback algorithm
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(0)
color = np.random.randint(0, 255, (200, 3))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
>>>>>>> 68c9ddf (ShiTomasi corner detection)

prev_time = 0 
while True:
    ret, frame = cap.read()
    if not ret:
        break
<<<<<<< HEAD


    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time


    cv2.putText(frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Live Camera Feed", frame)
=======
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    motion = np.zeros(2)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        motion += np.array([a - c, b - d])
        
    motion /= len(good_new)
>>>>>>> 68c9ddf (ShiTomasi corner detection)
    
    kalman.correct(np.array([[motion[0]], [motion[1]]], np.float32))
    predicted = kalman.predict()
    dx, dy = predicted[0][0], predicted[1][0]
    
    translation_matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    stabilized_frame = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(stabilized_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    
    img = cv2.add(stabilized_frame, mask)
    cv2.imshow('Stabilized Video', img)
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()