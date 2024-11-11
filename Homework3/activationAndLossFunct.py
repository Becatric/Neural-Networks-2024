import numpy as np

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
    return np.where(x > 0, 1, 0)

def softmax(logits):
    expLogits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return expLogits / np.sum(expLogits, axis=1, keepdims=True)

def loss(predictions, trueLabels):
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    return -np.mean(np.sum(trueLabels * np.log(predictions), axis=1))
