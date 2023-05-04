# Implement the liveness detection #
# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
parse = argparse.ArgumentParser()
parse.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
parse.add_argument("-l", "--lbe", type=str, required=True, help="path to label encoder")
parse.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detector")
parse.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability tp filter weak detection")
args = vars(parse.parse_args())

# Initialize the face detector #
# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder form disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
lbe = pickle.loads(open(args["lbe"], "rb").read())

# initialize the video stream and allow the camera sensor to warm-up
print("[INFO] starting video stream...")
videoStr = VideoStream(scr=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to have a maximum width of 600 pixels
    frame = videoStr.read()
    frame = imutils.resize(frame, width=600)

    # grab the frame dimensions and convert it to a blob
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # liveness detection with OpenCV and deep learning #
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x,y)-coordinate of the bounding box for the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the detected bounding box fall outside the dimensions of the stream
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(width, endX)
            endY = min(height, endY)

            # extract the face ROI and then preprocess it in the exact same manner as our training data
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (64, 64))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # pass the face ROI through the trained liveness detector model to determine if the face is "real" or "fake"
            predictions = model.predict(face)[0]
            j = np.argmax(predictions)
            label = lbe.classes_[j]

            # draw the label and bounding box on the frame
            label = "{}: {:.4f}".format(label, predictions[j])
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # show the output frame and wait for a key press
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the "q" key is pressed, break from the loop
    if key == ord("q"):
        break

# do the clean-up operations
cv2.destroyAllWindows()
videoStr.stop()
