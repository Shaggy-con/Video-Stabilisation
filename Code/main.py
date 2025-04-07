import cv2
import numpy as np

feature_params = dict(maxCorners=200, 
                      qualityLevel=0.01, 
                      minDistance=30, 
                      blockSize=3)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), 
                 maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Video capture
cap = cv2.VideoCapture(0)

# Create some random colors for visualization
color = np.random.randint(0, 255, (200, 3))

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial points to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing (for visualization purposes)
mask = np.zeros_like(old_frame)

# Kalman Filter initialization for smoother motion
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    # Estimate motion between frames
    motion = np.zeros(2)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        motion += np.array([a - c, b - d])
        
    motion /= len(good_new)
    
    # Apply Kalman filter to smooth motion
    kalman.correct(np.array([[motion[0]], [motion[1]]], np.float32))
    predicted = kalman.predict()
    
    # Adjust the frame based on predicted motion
    dx, dy = predicted[0], predicted[1]
    translation_matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    stabilized_frame = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
    
    # Update mask (for visualization)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(stabilized_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    
    # Overlay the original frame with the mask
    img = cv2.add(stabilized_frame, mask)
    
    # Display the stabilized frame
    cv2.imshow('Stabilized Video', img)
    
    # Update previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()