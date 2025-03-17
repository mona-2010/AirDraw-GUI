
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize MediaPipe Hand Tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize Camera
cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)

# Drawing Variables
drawing_color = (0, 0, 255)  # Default color: Red
canvas = np.zeros((hCam, wCam, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0

def clear_canvas():
    global canvas
    canvas = np.zeros((hCam, wCam, 3), dtype=np.uint8)

# Tkinter GUI
root = tk.Tk()
root.title("Air Drawing with Hand Tracking")
root.geometry("700x550")

# Video Label
video_label = tk.Label(root)
video_label.pack()

# Clear Button
tk.Button(root, text="Clear Screen", command=clear_canvas, font=("Arial", 12)).pack()

# Function to Process Video Frame
def update_frame():
    global prev_x, prev_y, canvas
    success, img = cap.read()
    if not success:
        return
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            if len(lmList) >= 8:
                x1, y1 = lmList[8][1], lmList[8][2]  # Index finger tip
                
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1
                
                # Draw on canvas
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), drawing_color, 5)
                prev_x, prev_y = x1, y1
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0  # Reset when no hand detected
    
    img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    video_label.imgtk = img
    video_label.configure(image=img)
    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()