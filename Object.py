import cv2
import numpy as np
import time

def create_background(cap,num_frames=30):
    print("Please move out of frame.!! Capturing background....")

    backgrounds=[]

    for i in range(num_frames):
        ret,frame = ret,frame=cap.read()
    
        if ret:
            backgrounds.append(frame)
        else:
            print(f"Warning: Could not read frame {i+1}/{num_frames}")
        time.sleep(0.1)

        if backgrounds:
            return np.median(backgrounds, axis=0).astype(np.uint8)
        else:
            raise ValueError("Could not capture any frame in background")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error")
    
    try:
        background = create_background(cap)
    except ValueError as e:
        print(f"Error: {e}")
        cap.release()
        return

    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 30])

    while True:
        ret,frame = cap.read()
        if not ret:
            print("Error: Can't read the frame!")
            time.sleep(1)
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt)>500:
                x,y,w,h = cv2.boundingRect(cnt)
                cx,cy = x+w//2 , y+h//2
                cv2.circle(frame, (cx,cy),5,(0,255,0),-1)
                print(f"Object center: ({cx},{cy})")
        
        cv2.imshow("Object detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    