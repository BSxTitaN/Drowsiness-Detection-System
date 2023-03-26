# Import the necessary packages
import datetime as dt
from imutils import face_utils
import matplotlib.pyplot as plt
from matplotlib import style
import imutils
import dlib
import time
import cv2
from playsound import playsound
import numpy as np
from scipy.spatial import distance as dist
import pandas as pd

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR



def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

    # check if C is greater than zero to avoid division by zero error
    if C > 0:
        # compute the mouth aspect ratio
        mar = (A + B) / (2.0 * C)

        # return the mouth aspect ratio
        return mar
    else:
        # return a default value of -1 if C is zero
        return -1




def detect_yawn(mouth):
    if  len(mouth) == 0:
        return None  # return None if the input is empty
    top_lip = mouth[50:53] + mouth[61:64]
    bottom_lip = mouth[56:59] + mouth[65:68]
    if not top_lip or not bottom_lip:
        return None  # return None if either lip list is empty
    top_mean = sum(top_lip) / len(top_lip)
    bottom_mean = sum(bottom_lip) / len(bottom_lip)
    yawn_dist = abs(top_mean[1] - bottom_mean[1])
    return yawn_dist


# Creating the dataset


# all eye  and mouth aspect ratio with time
ear_list = []
total_ear = []
mar_list = []
total_mar = []
ts = []
total_ts = []
# Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink
EAR_THRESHOLD = 0.3
# Declare another constant to hold the consecutive number of frames to consider for a blink
CONSECUTIVE_FRAMES = 20
# Another constant which will work as a threshold for MAR value
MAR_THRESHOLD = 14

# Initialize two counters
BLINK_COUNT = 0
FRAME_COUNT = 0

# Now, intialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
print("[INFO]Loading the predictor.....")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landamarks for the left and right eye respectively
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Now start the video stream and allow the camera to warm-up
print("[INFO]Loading Camera.....")
vs = cv2.VideoCapture(0)
time.sleep(2)
count_sleep = 0
count_yawn = 0
counter = 0
EYE_AR_CONSEC_FRAMES = 3
total_blink =0
# Now, loop over all the frames and detect the faces
while True:
    # Extract a frame
    ret,frame = vs.read()

    if frame is not None:
        frame = np.array(frame)

    cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    # Resize the frame
    frame = imutils.resize(frame, width=500)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    rects = detector(frame, 1)

    # Now loop over all the face detections and apply the predictor
    for rect in rects:
        shape = predictor(gray, rect)
        # Convert it to a (68, 2) size numpy array
        shape = face_utils.shape_to_np(shape)

        # Draw a rectangle over the detected face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Put a number
        cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend]
        mouth = shape[mstart:mend]
        # Compute the EAR for both the eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Take the average of both the EAR
        EAR = (leftEAR + rightEAR) / 2.0
        # live datawrite in csv
        ear_list.append(EAR)
        # print(ear_list)

        ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
        # Compute the convex hull for both the eyes and then visualize it
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # Draw the contours
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        MAR = mouth_aspect_ratio(mouth)
        mar_list.append(MAR / 10)
        # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place
        # Thus, count the number of frames for which the eye remains closed
        if EAR < EAR_THRESHOLD:
            counter += 1
        else:
            if counter >= EYE_AR_CONSEC_FRAMES:
                total_blink += 1
            counter = 0
        if EAR < EAR_THRESHOLD:
            FRAME_COUNT += 1

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                count_sleep += 1
                print(count_sleep)
                # Add the frame to the dataset ar a proof of drowsy driving
                cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            FRAME_COUNT = 0
        # cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check if the person is yawning
        if MAR > MAR_THRESHOLD:
            count_yawn += 1
            cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
            cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Add the frame to the dataset ar a proof of drowsy driving
            cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)

    # total data collection for plotting
    for i in ear_list:
        total_ear.append(i)
    for i in mar_list:
        total_mar.append(i)
    for i in ts:
        total_ts.append(i)
    # display the frame
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()

