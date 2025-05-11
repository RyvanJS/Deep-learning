import numpy as np
import load_dataset as ld

def main():
    x1, x2, x3, x4, y = ld.Load.get_data()

    X = np.column_stack((
        np.ones(x1.size),  # bias term
        x1,
        x2,
        x3
        # You can also add x4 here if you want
    ))

    Y = y.reshape(-1, 1)

    w = train(
        X, Y, iterations=1000, lr=0.001
    )

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, w):
    weighted_sum = np.dot(X, w)
    return sigmoid(weighted_sum)

def classify(X, w):
    return np.round(forward(X, w))

def loss(X, Y, w):
    predictions = forward(X, w)
    a = Y * np.log(predictions + 1e-15)  # add epsilon to avoid log(0)
    b = (1 - Y) * np.log(1 - predictions + 1e-15)
    return -np.average(a + b)

def gradient(X, Y, w):
    return (np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]).flatten()


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], ))
    for i in range(iterations):
        print("Iterasi %4d -> Loss: %.6f" % (i + 1, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

main()
