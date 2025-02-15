import cv2
import numpy as np

# Parameters for ShiTomasi corner detection (good features to track)
feature_params = dict(maxCorners=500, 
                      qualityLevel=0.01, 
                      minDistance=30, 
                      blockSize=3)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(21, 21), 
                 maxLevel=4, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Video capture
cap = cv2.VideoCapture(0)

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Failed to grab the first frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial points to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
if p0 is None:
    print("No features to track in the initial frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Kalman Filter initialization for smoother motion
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
kalman.statePre = np.zeros((4, 1), np.float32)
kalman.statePost = np.zeros((4, 1), np.float32)

# Initial transformation matrix (identity matrix)
last_transformation = np.eye(3, dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab the frame.")
        break
    
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Mask for tracking points that are within the frame
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) > 10:
            # Calculate transformation matrix using the good points
            transformation_matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
            
            if transformation_matrix is not None:
                # Convert 2x3 matrix to 3x3 matrix
                transformation_matrix = np.vstack([transformation_matrix, [0, 0, 1]])
                
                # Apply Kalman filter to smooth the transformation
                kalman.correct(np.array([transformation_matrix[0, 2], transformation_matrix[1, 2]], np.float32))
                predicted = kalman.predict()
                
                # Extract translation components
                dx, dy = predicted[0][0], predicted[1][0]
                
                # Adjust the transformation matrix based on predicted motion
                transformation_matrix[0, 2] = dx
                transformation_matrix[1, 2] = dy
                
                # Combine with the last transformation to reduce drift
                last_transformation = np.dot(last_transformation, transformation_matrix)
                
                # Apply the transformation to stabilize the frame
                stabilized_frame = cv2.warpAffine(frame, last_transformation[:2, :], (frame.shape[1], frame.shape[0]))
            else:
                stabilized_frame = frame
                print("Transformation matrix calculation failed.")
        else:
            stabilized_frame = frame
            print("Not enough good points for transformation.")
    else:
        stabilized_frame = frame
        print("Optical flow calculation failed.")

    # Combine original and stabilized frames side by side
    combined_frame = np.hstack((frame, stabilized_frame))
    
    # Draw tracking points
    if good_new is not None:
        for p in good_new:
            x, y = p.ravel()
            x, y = int(round(x)), int(round(y))  # Convert coordinates to integers
            cv2.circle(frame, (x, y), 5, color=(0, 255, 0), thickness=-1)

    # Display the frame with tracking points
    cv2.imshow('Tracked Points', frame)
    
    # Display the combined frame
    cv2.imshow('Original and Stabilized Video', combined_frame)
    
    # Update previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else p0
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
