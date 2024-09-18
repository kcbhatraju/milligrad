import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10),
])

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

epochs = 10
train_start = time.time()

model.compile(optimizer=optimizer, loss=loss)
model.fit(data, epochs=epochs, verbose=2)

train_end = time.time()
print(f"Train Time: {round(train_end-train_start, 3)} seconds")
print(f"Avg. Time per Epoch: {round((train_end-train_start)/epochs, 3)} seconds")

test_start = time.time()

outputs = model.predict(x_test, verbose=2)
pred = np.argmax(outputs, axis=1)
y_test = np.argmax(y_test, axis=1)
correct = np.sum(pred == y_test)

print(f"Test Accuracy: {correct/x_test.shape[0]}")

test_end = time.time()
print(f"Test Time: {round(test_end-test_start, 3)} seconds")

fig, axs = plt.subplots(2, 5)
for ax in axs.flat:
    idx = np.random.randint(0, x_test.shape[0])
    ax.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"{pred[idx]}")
    ax.axis("off")

plt.show()
