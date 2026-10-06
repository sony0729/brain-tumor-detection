"""Train the VGG16 brain tumor classifier and save it to backend/model/brain_tumor_vgg16.h5

Dataset layout (Kaggle "Brain MRI Images for Brain Tumor Detection"):
    brain_tumor_dataset/
        no/   *.jpg
        yes/  *.jpg

Run:  python train.py --data ./brain_tumor_dataset
"""
import argparse
import os

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="./brain_tumor_dataset")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--out", default="model/brain_tumor_vgg16.h5")
args = ap.parse_args()

images, labels = [], []
def list_images(root):
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(d, f)


for p in list_images(args.data):
    img = cv2.imread(p)
    if img is None:
        continue
    img = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)  # RGB, same as the API
    images.append(img)
    labels.append(p.split(os.path.sep)[-2])

images = np.array(images) / 255.0
lb = LabelBinarizer()
labels = to_categorical(lb.fit_transform(labels))
print("classes:", lb.classes_)  # ['no' 'yes'] -> index 0 = no tumor, 1 = tumor

train_X, test_X, train_Y, test_Y = train_test_split(
    images, labels, test_size=0.10, random_state=42, stratify=labels
)
aug = ImageDataGenerator(fill_mode="nearest", rotation_range=15)

base = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
for layer in base.layers:
    layer.trainable = False
x = AveragePooling2D(pool_size=(4, 4))(base.output)
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(2, activation="softmax")(x)
model = Model(inputs=base.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])

bs = 8
model.fit(
    aug.flow(train_X, train_Y, batch_size=bs),
    validation_data=(test_X, test_Y),
    epochs=args.epochs,
)

pred = np.argmax(model.predict(test_X, batch_size=bs), axis=1)
true = np.argmax(test_Y, axis=1)
print(classification_report(true, pred, target_names=lb.classes_))
print(confusion_matrix(true, pred))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
model.save(args.out)
print("saved ->", args.out)
