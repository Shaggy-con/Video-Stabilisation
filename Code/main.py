import cv2
import time
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

prev_time = 0 
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break


    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time


    cv2.putText(frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Live Camera Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()