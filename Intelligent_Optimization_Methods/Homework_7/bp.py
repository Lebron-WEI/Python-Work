import numpy as np

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def bp(vector, target, weights, biases, lr):
    hidden = sigmoid(np.matmul(vector, weights[0]) + biases[0])
    out = sigmoid(np.matmul(hidden, weights[1]) + biases[1])

    erro = out * (1 - out) * (target - out)
    errhidden = hidden * (1 - hidden) * np.matmul(erro, weights[1].T)
    weights[0] += lr * np.matmul(vector.T, errhidden)
    weights[1] += lr * np.matmul(hidden.T, erro)
    biases[0] += lr * errhidden
    biases[1] += lr * erro
    return weights, biases

vector = np.array([1,0,1,0,0,1,0,0,0]).reshape(1,-1)
print(vector)
target = np.array([1,0]).reshape(1,-1)
weights = [np.array([[0.5]*3]*9), np.array([[0.5]*2]*3)]
print(weights)
biases = [np.array([0.0,0.0,0.0]).reshape(1,-1), np.array([0.0,0.0]).reshape(1,-1)]
w, b = bp(vector, target, weights, biases, 1)
print(w)
print(b)