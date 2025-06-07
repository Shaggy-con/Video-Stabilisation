import numpy as np
import cv2
from filterpy.kalman import KalmanFilter

def movingAverage(curve, radius):
    if len(curve.shape) != 1:
        raise ValueError("Input curve must be a 1D array.")
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), mode='edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth(trajectory):
    if len(trajectory.shape) != 2 or trajectory.shape[1] != 3:
        raise ValueError("Trajectory must be a 2D array with 3 columns.")
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

# Parameters
SMOOTHING_RADIUS = 50

# Initialize Kalman filter
kf = KalmanFilter(dim_x=3, dim_z=3)  # 3 states (dx, dy, da), 3 measurements (dx, dy, da)
kf.F = np.eye(3)  # State transition matrix
kf.H = np.eye(3)  # Measurement matrix
kf.P *= 1000.  # Initial uncertainty
kf.R = np.eye(3) * 0.1  # Measurement noise
kf.Q = np.eye(3) * 0.01  # Process noise

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Read the first frame
ret, prev = cap.read()
if not ret:
    print("Failed to grab the first frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Get frame width and height
h, w, _ = prev.shape

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Initialize transformation array
transforms = []

while True:
    # Read next frame
    ret, curr = cap.read()
    if not ret:
        print("Failed to grab the frame.")
        break

    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Detect feature points
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    if prev_pts is None:
        print("No feature points detected.")
        continue

    # Calculate optical flow (track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Filter valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    if len(prev_pts) > 0:
        # Estimate transformation
        m, _ = cv2.estimateAffine2D(prev_pts, curr_pts, method=cv2.RANSAC)

        if m is not None:
            # Extract transformation parameters
            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])

            # Update Kalman filter
            kf.predict()
            kf.update([dx, dy, da])

            # Get smoothed transformation from Kalman filter
            dx, dy, da = kf.x.flatten()

            # Store the transformation
            transforms.append([dx, dy, da])
        else:
            # Use last valid transformation if estimation fails
            if len(transforms) > 0:
                transforms.append(transforms[-1])
            else:
                transforms.append([0, 0, 0])
    else:
        # Use last valid transformation if no points are tracked
        if len(transforms) > 0:
            transforms.append(transforms[-1])
        else:
            transforms.append([0, 0, 0])

    # Compute trajectory
    trajectory = np.cumsum(transforms, axis=0)

    # Debugging: Print shape of trajectory
    print("Trajectory shape before smoothing:", trajectory.shape)

    # Reshape trajectory if necessary
    trajectory = trajectory.reshape(-1, 3)

    print("Trajectory shape after reshaping:", trajectory.shape)

    # Smooth trajectory
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = np.array(transforms) + difference

    # Apply transformations to current frame
    if len(transforms_smooth) > 0:
        dx, dy, da = transforms_smooth[-1]
        
        # Ensure that dx, dy, and da are scalars
        dx = float(dx)
        dy = float(dy)
        da = float(da)

        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine transformation
        frame_stabilized = cv2.warpAffine(curr, m, (w, h))
        frame_stabilized = fixBorder(frame_stabilized)
    else:
        frame_stabilized = curr

    # Create mask for tracking points
    tracking_mask = np.zeros_like(curr)
    if prev_pts is not None:
        for p in curr_pts:
            x, y = p.ravel()
            x, y = int(round(x)), int(round(y))  
            cv2.circle(tracking_mask, (x, y), 5, color=(0, 255, 0), thickness=-1)

    # Combine original frame and tracking mask
    frame_with_tracking_points = cv2.add(curr, tracking_mask)

    # Combine original frame and stabilized frame side by side
    combined_frame = cv2.hconcat([curr, frame_stabilized])

    # Display the frames
    cv2.imshow("Original and Stabilized Video", combined_frame)
    cv2.imshow("Tracked Points", frame_with_tracking_points)

    # Update previous frame and previous points
    prev_gray = curr_gray
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()


