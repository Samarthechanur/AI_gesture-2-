import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    if tf.shape(image).shape[0] == 2:
        image = tf.expand_dims(image, -1)  # Ensure channel dimension
    image = tf.transpose(image, [1,0,2])  # Correct rotation
    return image, label

def filter_valid(image, label):
    return tf.less(label, 36)

print("Loading EMNIST balanced dataset for augmentation...")

train_ds = tfds.load('emnist/balanced', split='train', as_supervised=True)
test_ds = tfds.load('emnist/balanced', split='test', as_supervised=True)

train_ds = train_ds.filter(filter_valid).map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.filter(filter_valid).map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

# Convert the dataset to numpy arrays for ImageDataGenerator
def tfds_to_numpy(ds):
    images = []
    labels = []
    for img_batch, lbl_batch in ds.unbatch().take(10000):  # Limit for memory; increase as possible
        images.append(img_batch.numpy())
        labels.append(lbl_batch.numpy())
    return np.array(images), np.array(labels)

train_images, train_labels = tfds_to_numpy(train_ds)
test_images, test_labels = tfds_to_numpy(test_ds)

# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)

# Build a deeper CNN with batch norm and dropout (same arch as before)
model = models.Sequential([
    layers.Input(shape=(28,28,1)),

    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(36, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
print("Training with data augmentation...")

# Fit model using augmented data
model.fit(
    datagen.flow(train_images, train_labels, batch_size=128),
    steps_per_epoch=len(train_images)//128,
    epochs=20,
    validation_data=(test_images, test_labels)
)

model.save("model/hand_gesture_model.h5")
print("Augmented and fine-tuned model saved!")