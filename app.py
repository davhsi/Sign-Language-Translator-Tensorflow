import os
import customtkinter as ctk
import csv
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
from model import KeyPointClassifier
import itertools
import copy
from datetime import datetime

# Function to calculate the landmark points from an image
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# Function to preprocess landmark data
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    return list(map(lambda n: n / max_value, temp_landmark_list))

# Get the base path for the data files
base_path = os.path.dirname(__file__)

# Load the KeyPointClassifier model
model_path = os.path.join(base_path, 'model', 'keypoint_classifier', 'model.tflite')
keypoint_classifier = KeyPointClassifier(model_path)  # Make sure your KeyPointClassifier accepts this path

# Read labels from a CSV file
label_path = os.path.join(base_path, 'model', 'keypoint_classifier', 'label.csv')
with open(label_path, encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

# Set the appearance mode and color theme for the custom tkinter library
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Create the main window
window = ctk.CTk()
window.geometry('1080x1080')
window.title("DAV Sign Language Predictor")
prev = ""

# Function to open the camera and perform hand gesture recognition
def open_camera1():
    global prev
    width, height = 1000, 1000
    with mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as hands:
        _, frame = vid.read()
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        opencv_image = cv2.resize(opencv_image, (width, height))
        processFrames = hands.process(opencv_image)
        
        if processFrames.multi_hand_landmarks:
            for lm in processFrames.multi_hand_landmarks:
                mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)

                landmark_list = calc_landmark_list(frame, lm)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                cur = keypoint_classifier_labels[hand_sign_id]

                if cur == prev:
                    letter.configure(text=cur)
                elif cur:
                    prev = cur
                    Sentence.insert(tk.END, cur + " ")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frame = cv2.flip(frame, 1)
        captured_image = Image.fromarray(frame)
        my_image = ctk.CTkImage(dark_image=captured_image, size=(340, 335))
        video_label.configure(image=my_image)
        video_label.after(10, open_camera1)

# Initialize the video capture
vid = cv2.VideoCapture(0)
mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils

# Create a colorful theme
window.configure(bg='#2E4053')  # Dark steel blue

# Create the title label with new vibrant color
title_font = ctk.CTkFont(family='Consolas', weight='bold', size=30)
title_label = ctk.CTkLabel(window, text="DAV HAND SIGNS RECOGNITION", fg_color='orange', text_color='white', height=50, font=title_font, corner_radius=8)
title_label.pack(side=ctk.TOP, fill=ctk.X, pady=(10, 10), padx=(10, 10))

# Create the main frame with a colorful background and centered layout
main_frame = ctk.CTkFrame(window, bg_color='#5D6D7E', height=800, corner_radius=10)
main_frame.pack(fill=ctk.BOTH, expand=True, padx=(200, 200), pady=(10, 10))  # Adjust padding to center the frame

# Create the left frame for video feed and center it in the main frame
left_frame = ctk.CTkFrame(main_frame, bg_color='#85C1E9', width=400, height=600, corner_radius=12)
left_frame.pack(side=ctk.LEFT, padx=(10, 10), pady=(10, 10), expand=True, fill=tk.BOTH)  # Added expand and fill to balance center

# Video display area
video_frame = ctk.CTkFrame(left_frame, height=500, corner_radius=12)
video_frame.pack(padx=(10, 10), pady=(10, 10), expand=True)
video_label = ctk.CTkLabel(video_frame, text='', height=500, corner_radius=12)
video_label.pack(fill=ctk.BOTH, padx=(10, 10), pady=(10, 10))

# Start button for camera feed
Camera_feed_start = ctk.CTkButton(left_frame, text="OPEN CAMERA", height=50, width=250, corner_radius=12, fg_color="green", command=open_camera1)
Camera_feed_start.pack(pady=(10, 10))

# Create the right frame for recognized letters
right_frame = ctk.CTkFrame(main_frame, bg_color='#F1948A', width=400, height=600, corner_radius=12)
right_frame.pack(side=ctk.RIGHT, padx=(10, 10), pady=(10, 10), expand=True, fill=tk.BOTH)  # Balance with expand and fill

# Display recognized hand sign
letter_font = ctk.CTkFont(family='Consolas', weight='bold', size=180)
letter = ctk.CTkLabel(right_frame, text='', font=letter_font, fg_color='white', text_color='black', justify=ctk.CENTER, corner_radius=12)
letter.pack(fill=ctk.BOTH, padx=(10, 10), pady=(10, 10))

# Create the textbox to display the predicted text and center it
textbox_frame = ctk.CTkFrame(window, height=150, corner_radius=12, fg_color="#D5DBDB")
textbox_frame.pack(fill=ctk.X, padx=(200, 200), pady=(10, 10))  # Adjusted padding to center the textbox

Sentence = ctk.CTkTextbox(textbox_frame, height=140, font=("Consolas", 20), corner_radius=10)
Sentence.pack(fill=ctk.BOTH, expand=True, padx=(10, 10), pady=(10, 10))

# Start the tkinter main loop
window.mainloop()
