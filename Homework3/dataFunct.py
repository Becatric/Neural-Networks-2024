import numpy as np
from torchvision.datasets import MNIST

def downloadMnist(isTrain):
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten(), download=True, train=isTrain)
    data = []
    labels = []
    for image, label in dataset:
        data.append(image)
        labels.append(label)
    return np.array(data), np.array(labels)

def encode(labels, numClasses):
    matrix = np.zeros((labels.size, numClasses))
    for i, label in enumerate(labels):
        matrix[i, label] = 1
    return matrix

def createBatches(features, labels, batchSize):
    numBatches = features.shape[0] // batchSize
    batches = []
    for i in range(numBatches):
        batchFeatures = features[i * batchSize: (i + 1) * batchSize]
        batchLabels = labels[i * batchSize: (i + 1) * batchSize]
        batches.append((batchFeatures, batchLabels))
    return batches
