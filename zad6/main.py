import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui as keyboard

"""
The main goal of the program is to control youtube player,
by hand gestures captured in your camera.
List of supported gestures:
#peace sign - leave program 
#thumbs up - volume up
#thumbs down - volume down
#okay - mute/unMute
#live long - stop/play
#rock - back 5s
#fist - forward 5s

Warning:
Program waits a little when you use gestures like #okay or #live long
to prevent the app to stop/play or mute/unmute immediately 

You need to have installed packages: tensorflow, pyautogui, mediapipe, numpy, opencv-python
Authors: Bartosz Kamiński, Michał Czerwiak
"""

"""
Set up mediapipe settings
"""
mpSolutionHands = mp.solutions.hands
mpHands = mpSolutionHands.Hands(max_num_hands=1, min_detection_confidence=0.9)
mpDraw = mp.solutions.drawing_utils

"""
Load TensorFlow pre-trained model 
"""
handModel = load_model('mp_hand_gesture_data')

"""
Load control gesture names
"""
f = open('gesture.names', 'r')
gestureNames = f.read().split('\n')
f.close()
print(gestureNames)

"""
Initialize the webcam, set desired webcam resolution(You can use 1920, 1080 for example)
"""
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
"""
Implementing counter to stop gesture recognition for certain gestures
"""
counter = 0
while True:
    _, frame = cap.read()

    x, y, c = frame.shape

    """
    Flip the frame vertically
    """
    frame = cv2.flip(frame, 1)
    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    """
    Get prediction of hand landmarks
    """
    result = mpHands.process(frameRgb)

    gestureNameId = -1

    if result.multi_hand_landmarks and counter > -1:
        landMarks = []
        for handLandMarks in result.multi_hand_landmarks:
            for lm in handLandMarks.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landMarks.append([lmx, lmy])

            """
            Predict hand gestures captured on the camera
            """
            prediction = handModel.predict([landMarks])

            """
            Draw landmarks on frames
            """
            mpDraw.draw_landmarks(frame, handLandMarks, mpSolutionHands.HAND_CONNECTIONS)

            """
            Get index of predicted gesture
            """
            gestureNameId = np.argmax(prediction)

    """
    Type the captured prediction on the frame
    """
    cv2.putText(frame, gestureNames[gestureNameId], (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                2, (255, 255, 0), 2, cv2.LINE_AA)
    """
    Press the appropriate button based on the prediction (control youtube player)
    """
    if gestureNameId == 0:
        keyboard.press("m")
        counter = -30
    elif gestureNameId == 2:
        keyboard.press("up")
    elif gestureNameId == 3:
        keyboard.press("down")
    elif gestureNameId == 7:
        keyboard.press("space")
        counter = -30
    elif gestureNameId == 6:
        keyboard.press("left")
    elif gestureNameId == 8:
        keyboard.press("right")
    else:
        pass

    """
    Show the Gestures Handler Dialog
    """
    cv2.imshow("Gesture Handler", frame)

    """
    Quit the program if hit 'q' on keyboard
    """

    if cv2.waitKey(1) == ord('q'):
        break
    if counter < 0:
        counter += 1

"""
Release the webcam and quit all windows
"""
cap.release()
cv2.destroyAllWindows()
