import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
import pandas as pd

DATASET_PATH = 'dataset/'
IMG_SIZE = (128, 128)

labels_df = pd.read_csv(os.path.join(DATASET_PATH, 'labels.csv'))

def load_images(path, img_names, folder):
    images = []
    for name in img_names:
        img_path = os.path.join(DATASET_PATH, folder, name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        images.append(img)
    return np.array(images)

temp_images = load_images(DATASET_PATH, labels_df['image_name'], 'temperature')
normal_images = load_images(DATASET_PATH, labels_df['image_name'], 'normal')

label_map = {'low': 0, 'medium': 1, 'high': 2}
labels = np.array([label_map[l] for l in labels_df['label']])
labels = to_categorical(labels, 3)

X_temp_train, X_temp_test, X_norm_train, X_norm_test, y_train, y_test = train_test_split(
    temp_images, normal_images, labels, test_size=0.2, random_state=42
)

def create_cnn_branch(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3)
    ])
    return model

input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
temp_branch = create_cnn_branch(input_shape)
normal_branch = create_cnn_branch(input_shape)
combined_input = layers.concatenate([temp_branch.output, normal_branch.output])
x = layers.Dense(128, activation='relu')(combined_input)
x = layers.Dropout(0.4)(x)
output = layers.Dense(3, activation='softmax')(x)

model = models.Model(inputs=[temp_branch.input, normal_branch.input], outputs=output)
model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit([X_temp_train, X_norm_train], y_train, validation_split=0.2, epochs=20, batch_size=16)
loss, acc = model.evaluate([X_temp_test, X_norm_test], y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

def predict_uhi(temp_img_path, normal_img_path):
    temp_img = cv2.imread(temp_img_path)
    normal_img = cv2.imread(normal_img_path)
    temp_img = cv2.resize(temp_img, IMG_SIZE)/255.0
    normal_img = cv2.resize(normal_img, IMG_SIZE)/255.0
    pred = model.predict([[temp_img], [normal_img]])
    classes = ['Low UHI', 'Medium UHI', 'High UHI']
    print("Predicted UHI Level:", classes[np.argmax(pred)])
