import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, [1,0,2])  # Rotate to correct orientation
    image = tf.expand_dims(image, -1)
    return image, label

print("Loading EMNIST byclass dataset...")
train_ds = tfds.load('emnist/byclass', split='train', as_supervised=True)
test_ds = tfds.load('emnist/byclass', split='test', as_supervised=True)

train_ds = train_ds.map(preprocess).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(62, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
print("Training for 5 epochs...")
model.fit(train_ds, epochs=5, validation_data=test_ds)

model.save("model/hand_gesture_model")  # NOTE: save as TF native format, no .h5 extension
print("Model saved successfully!")
