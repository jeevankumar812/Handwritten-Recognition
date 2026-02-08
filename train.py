import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

# ---------------------------
IMG = 64
DATA = "dataset/characters"
CLASSES = sorted(os.listdir(DATA))   # ['a','b']
# ---------------------------

X = []
y = []

# LOAD DATA
for i, cls in enumerate(CLASSES):
    folder = os.path.join(DATA, cls)
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        image = cv2.imread(path, 0)
        image = cv2.threshold(
            image, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        image = cv2.resize(image, (IMG, IMG))
        X.append(image)
        y.append(i)

X = np.array(X)/255.0
X = X.reshape(-1, IMG, IMG, 1)
y = to_categorical(y)

# AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# MODEL
model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(64,64,1)),
    MaxPooling2D(),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128,activation="relu"),
    Dropout(0.3),

    Dense(len(CLASSES),activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# TRAIN
history = model.fit(
    datagen.flow(X,y,batch_size=8),
    epochs=30
)

# SAVE MODEL
model.save("model/model.h5")
pickle.dump(CLASSES, open("model/labels.pkl","wb"))

# ---------------------------
# GRAPH 1 - Accuracy
# ---------------------------
plt.figure()
plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# ---------------------------
# GRAPH 2 - Loss
# ---------------------------
plt.figure()
plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ---------------------------
# GRAPH 3 - Accuracy Trend
# ---------------------------
plt.figure()
plt.plot(history.history["accuracy"])
plt.title("Accuracy Trend")
plt.show()

# ---------------------------
# GRAPH 4 - Loss Trend
# ---------------------------
plt.figure()
plt.plot(history.history["loss"])
plt.title("Loss Trend")
plt.show()

# ---------------------------
# GRAPH 5 - Confusion Matrix
# ---------------------------
pred = model.predict(X)
y_pred = np.argmax(pred,axis=1)
y_true = np.argmax(y,axis=1)

cm = confusion_matrix(y_true,y_pred)

plt.figure()
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Training Completed Successfully")
