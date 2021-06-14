# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

# batch size
epochs=10
#batch size
bs=32

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
directory='Images'
categories=["helmet","no_helmet"]

data=[]
labels=[]

for category in categories:
  path=os.path.join(directory,category)
  for img in os.listdir(path):
    img_path=os.path.join(path,img)
    image=load_img(img_path,target_size=(224,224))
    image=img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(category)
# load the input image (224x224) and preprocess i
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)
# convert the data and labels to NumPy arrays
data=np.array(data,dtype="float32")
labels=np.array(labels)


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX,testX,trainY,testY) = train_test_split(data, labels,
                                                    test_size=0.20,
                                                    random_state=0,
                                                    stratify=labels)

# construct the training image generator for data augmentation
aug=ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# load the MobileNetV2 network, ensuring the head FC layer sets are
baseModel=MobileNetV2(weights="imagenet",
                      include_top=False,
                      input_tensor=Input(shape=(224,224,3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name="flatten")(headModel)
headModel=Dense(120,activation="relu")(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model=Model(inputs=baseModel.input,outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable=False

# compile our model
print("[INFO] compiling model...")
opt=Adam(learning_rate=0.001)
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H=model.fit(
    aug.flow(trainX,trainY,batch_size=bs),
    steps_per_epoch = len(trainX)//bs,
    validation_data = (testX,testY),
    validation_steps = len(testX)//bs,
    epochs=epochs
)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs=model.predict(testX,batch_size=bs)

model.save("helmet.model",save_format="h5")
print("Model saved!")


# plot the training loss and accuracy
plt.plot(H.history["loss"], label="train_loss",color="red")
plt.plot(H.history["val_loss"], label="val_loss",color="blue")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.show()

plt.plot(H.history["accuracy"], label="train_acc",color="red")
plt.plot(H.history["val_accuracy"], label="val_acc",color="blue")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.show()
