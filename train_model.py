import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

def preprocess(image, label):
    # If image shape is [28,28], add channel dim
    image = tf.cast(image, tf.float32) / 255.0
    if tf.shape(image).shape[0] == 2:  # If shape=(28,28)
        image = tf.expand_dims(image, -1)
    # EMNIST images are transposed by default, so always transpose [28,28,1] -> [28,28,1]
    # To ensure correctness, always just transpose the first two dimensions
    image = tf.transpose(image, [1, 0, 2])
    return image, label

def filter_valid(image, label):
    # Only keep 0-9 and A-Z (labels 0-35)
    return tf.less(label, 36)

print("Loading EMNIST balanced dataset...")
train = tfds.load('emnist/balanced', split='train', as_supervised=True)
test = tfds.load('emnist/balanced', split='test', as_supervised=True)

train = train.filter(filter_valid).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(128).prefetch(tf.data.AUTOTUNE)
test = test.filter(filter_valid).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(128).prefetch(tf.data.AUTOTUNE)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(36, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
print("Training model...")
model.fit(train, epochs=10, validation_data=test)
model.save("model/hand_gesture_model.h5")
print("Model saved!")
