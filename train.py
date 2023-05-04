# Train the network #
# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from liveness_CNN.livenessNet_ResNet18 import ResNet18
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
parse = argparse.ArgumentParser()
parse.add_argument("-d", "--dataset", required=True, help="path to input dataset")
parse.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
parse.add_argument("-l", "--lbe", type=str, required=True, help="path to label encoder")
parse.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(parse.parse_args())

# Perform a number of initializations and build the data #
# initialize the initial learning rate, batch size and the number of epochs to train for
Init_LR = 1e-4
BS = 8
Epochs = 50

# grab the list of images in our dataset directory, then initialize the list of data (images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over all image paths
for imagePath in imagePaths:
    # extract the class label from the filename, load the image and resize it to be a fixed 32x32 pixels (ignoring AR)
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))

    # update the data and label lists respectively
    data.append(image)
    labels.append(label)

# convert the data into a Numpy array, then preprocess it by scaling all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then use one-hot encoding on them
lbe = LabelEncoder()
labels = lbe.fit_transform(labels)
labels = to_categorical(labels, 2)

# partition the data into training and testing splits using 75% and 25% of the data respectively
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Initialize the data argumentation object and compile+train the face model #
# construct the training image generator for data argumentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=Init_LR, decay=Init_LR / Epochs)
model = ResNet18((64, 64, 3), len(lbe.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(Epochs))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // BS, epochs=Epochs)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lbe.classes_))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"], save_format="h5")

# save the label encoder to disk
f = open(args["lbe"], "wb")
f.write(pickle.dumps(lbe))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, Epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, Epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, Epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, Epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig(args["plot"])
