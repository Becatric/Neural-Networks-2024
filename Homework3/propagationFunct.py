import numpy as np
from activationAndLossFunct import relu, reluDerivative, softmax

def forwardPropagation(batchFeatures, weightsHidden, biasHidden, weightsOutput, biasOutput, dropoutRate=0.0):
    hiddenInput = np.dot(batchFeatures, weightsHidden) + biasHidden
    hiddenOutput = relu(hiddenInput)

    if dropoutRate > 0:
        mask = np.random.rand(*hiddenOutput.shape) > dropoutRate
        hiddenOutput *= mask
        hiddenOutput /= (1.0 - dropoutRate)

    outputInput = np.dot(hiddenOutput, weightsOutput) + biasOutput
    probabilities = softmax(outputInput)

    cache = {
        'batchFeatures': batchFeatures,
        'hiddenInput': hiddenInput,
        'hiddenOutput': hiddenOutput,
        'mask': mask if dropoutRate > 0 else None,
        'outputInput': outputInput,
        'probabilities': probabilities
    }
    return probabilities, cache


def backwardPropagation(batchLabels, cache, weightsHidden, biasHidden, weightsOutput, biasOutput, learningRate, dropoutRate=0.0):
    numberOfSamples = batchLabels.shape[0]
    differenceOutput = cache['probabilities'] - batchLabels
    gradientWeightsOutput = np.dot(cache['hiddenOutput'].T, differenceOutput) / numberOfSamples
    gradientBiasOutput = np.sum(differenceOutput, axis=0, keepdims=True) / numberOfSamples
    differenceHidden = np.dot(differenceOutput, weightsOutput.T) * reluDerivative(cache['hiddenInput'])

    if dropoutRate > 0 and cache['mask'] is not None:
        differenceHidden *= cache['mask']
        differenceHidden /= (1.0 - dropoutRate)

    gradientWeightsHidden = np.dot(cache['batchFeatures'].T, differenceHidden) / numberOfSamples
    gradientBiasHidden = np.sum(differenceHidden, axis=0, keepdims=True) / numberOfSamples

    weightsOutput = weightsOutput - learningRate * gradientWeightsOutput
    biasOutput = biasOutput - learningRate * gradientBiasOutput
    weightsHidden = weightsHidden - learningRate * gradientWeightsHidden
    biasHidden = biasHidden - learningRate * gradientBiasHidden

    return weightsHidden, biasHidden, weightsOutput, biasOutput
