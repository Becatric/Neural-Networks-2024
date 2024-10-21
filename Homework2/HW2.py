import numpy as np
from torchvision.datasets import MNIST

def downloadMnist(is_train):
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten(), download=True, train=is_train)

    data = []
    labels = []

    for image, label in dataset:
        data.append(image)
        labels.append(label)

    return np.array(data), np.array(labels)


def encode(labels, num_classes):
    matrix = np.zeros((labels.size, num_classes))
    for i, label in enumerate(labels):
        matrix[i, label] = 1
    return matrix


def createBatches(features, labels, batchSize):
    nrBatches = features.shape[0] // batchSize
    batches = []
    for i in range(nrBatches):
        batch_features = features[i * batchSize: (i + 1) * batchSize]
        batch_labels = labels[i * batchSize: (i + 1) * batchSize]
        batches.append((batch_features, batch_labels))
    return batches


def softmax(logits):
    exp_logits = np.exp(logits )
    # - np.max(logits, axis=1, keepdims=True)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def forward_propagation(batch_features, weights, biases):
    logits = np.dot(batch_features, weights) + biases
    probabilities = softmax(logits)
    return probabilities


def loss(predictions, true_labels):
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    loss = -np.mean(np.sum(true_labels * np.log(predictions), axis=1))
    return loss

def backward_propagation(batch_features, batch_labels, predictions, weights, biases, learning_rate):
    m = batch_features.shape[0]
    dZ = batch_labels - predictions
    dW = np.dot(batch_features.T, dZ) / m
    dB = np.sum(dZ, axis=0, keepdims=True) / m
    weights += learning_rate * dW
    biases += learning_rate * dB
    return weights, biases


def train_model(train_batches, weights, biases, learning_rate, epochs):
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_features, batch_labels in train_batches:
            predictions = forward_propagation(batch_features, weights, biases)
            batch_loss = loss(predictions, batch_labels)
            epoch_loss += batch_loss
            weights, biases = backward_propagation(batch_features, batch_labels, predictions, weights, biases, learning_rate)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_batches)}")
    return weights, biases


def evaluate_model(test_features, test_labels, weights, biases):
    predictions = forward_propagation(test_features, weights, biases)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    accuracy = np.mean(predicted_labels == true_labels) * 100
    return accuracy


num_classes = 10
batch_size = 100
np.random.seed(0)
numInputs = 784
numClasses = 10
weight = np.random.randn(numInputs, numClasses) * 0.01
bias = np.zeros((1, num_classes))
learning_rate = 0.001
epochs = 2000

trainFeature, trainLabel = downloadMnist(True)
testFeature, testLabel = downloadMnist(False)

trainFeature = trainFeature / 255.0
testFeature = testFeature / 255.0

trainLabelEncoded = encode(trainLabel.flatten(), num_classes)
testLabelEncoded = encode(testLabel.flatten(), num_classes)

train_batches = createBatches(trainFeature, trainLabelEncoded, batch_size)
trained_weights, trained_biases = train_model(train_batches, weight, bias, learning_rate, epochs)

test_accuracy = evaluate_model(testFeature, testLabelEncoded, trained_weights, trained_biases)
print(f"Test Accuracy: {test_accuracy:.2f}%")

#
# print("Training data shape:", trainFeature.shape)
# print("Testing data shape:", testFeature.shape)

