# Generate real and fake samples into the Dataset #

# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
parse = argparse.ArgumentParser()
parse.add_argument("-i", "--input", type=str, required=True, help="path to input video")
parse.add_argument("-o", "--output", type=str, required=True, help="path to output directory of cropped faces")
parse.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detection")
parse.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
parse.add_argument("-s", "--skip", type=int, default=16, help="# of frames to skip before applying face detection")
args = vars(parse.parse_args())

# load our serialized face detector from disk
print("[INFO] loading the face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# open a pointer to the video file stream and initialize the total number of frames read and saved thus far
videoStr = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# loop over frames from the video file stream
while True:
    # grab the frame from the file
    (grabbed, frame) = videoStr.read()

    # if the frame was not grabbed, then the video stream have reached to the end
    if not grabbed:
        break

    # increase the total number of frames read thus far using 'read' counter
    read += 1

    # check if this frame should be processed
    if read & args["skip"] != 0:
        continue

    # grab the frame dimensions and construct a blob from the frame (image preprocessing)
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # ensure at least one face was found in the grabbed frame
    if len(detections) > 0:
        #  we assume that each image has only 1 face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

    # ensure that the detection with the largest probability also means our minimum probability test
    # use this method can filter out weak detections
    if confidence > args["confidence"]:
        # compute the (x,y)-coordinates of the bounding box for the face and extract the face ROI
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (startX, startY, endX, endY) = box.astype("int")
        face = frame[startY:endY, startX:endX]

        # write the frame to disk
        p = os.path.sep.join([args["output"], "{}.png".format(saved)])
        cv2.imwrite(p, face)
        saved += 1
        print("[INFO saved {} to disk".format(p))

# do the clean-up operations
videoStr.release()
cv2.destroyAllWindows()
