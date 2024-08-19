import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import milligrad as mg

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

data = mg.data.DataLoader(x_train, y_train, batch_size=128, shuffle=True)

model = mg.models.Sequential([
    mg.layers.Dense(28*28, 64, activation="relu"),
    mg.layers.Dense(64, 10),
])

loss = mg.losses.CrossentropyLoss(from_logits=True)
optimizer = mg.optimizers.Adam(model.parameters(), lr=0.001)

epochs = 10
train_start = time.time()

model.train()
for epoch in range(epochs):
    for x_batch, y_batch in data:
        outputs = model(x_batch)
        cost = loss(y_batch, outputs)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    
    print(f"Epoch: {epoch+1} Cost: {cost.item()}")

train_end = time.time()
print(f"Train Time: {round(train_end-train_start, 3)} seconds")
print(f"Avg. Time per Epoch: {round((train_end-train_start)/epochs, 3)} seconds")

test_start = time.time()

model.eval()

correct = 0
with mg.no_grad():
    outputs = model(x_test)
    pred = np.argmax(outputs.item(), axis=1)
    y_test = np.argmax(y_test, axis=1)
    correct += np.sum(pred == y_test)

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
