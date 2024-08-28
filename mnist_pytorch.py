import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

train_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

x_train = train_data.data.numpy()
y_train = train_data.targets.numpy()

x_test = test_data.data.numpy()
y_test = test_data.targets.numpy()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

train_data = torch.utils.data.DataLoader(
    dataset=torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
    batch_size=128,
    shuffle=False,
)

model = torch.nn.Sequential(
    torch.nn.Linear(28*28, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 10),
)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_start = time.time()

model.train()

for epoch in range(epochs):
    for x_batch, y_batch in train_data:
        outputs = model(x_batch.float())
        cost = loss(outputs, y_batch)

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
with torch.no_grad():
    outputs = model(torch.tensor(x_test).float())
    pred = np.argmax(outputs.numpy(), axis=1)
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
