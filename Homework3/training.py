import numpy as np
from propagationFunct import forwardPropagation, backwardPropagation
from activationAndLossFunct import loss

def trainModel(trainBatches, weightsHidden, biasHidden, weightsOutput, biasOutput, learningRate, epochs, dropoutRate=0.0):
    for epoch in range(epochs):
        epochLoss = 0
        for batchFeatures, batchLabels in trainBatches:
            predictions, cache = forwardPropagation(batchFeatures, weightsHidden, biasHidden, weightsOutput, biasOutput, dropoutRate)
            batchLoss = loss(predictions, batchLabels)
            epochLoss += batchLoss

            weightsHidden, biasHidden, weightsOutput, biasOutput = backwardPropagation(
                batchLabels, cache, weightsHidden, biasHidden, weightsOutput, biasOutput, learningRate, dropoutRate
            )

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epochLoss / len(trainBatches):.4f}")
    return weightsHidden, biasHidden, weightsOutput, biasOutput

def evaluateModel(testFeatures, testLabels, weightsHidden, biasHidden, weightsOutput, biasOutput):
    predictions, _ = forwardPropagation(testFeatures, weightsHidden, biasHidden, weightsOutput, biasOutput)
    predictedLabels = np.argmax(predictions, axis=1)
    trueLabels = np.argmax(testLabels, axis=1)
    accuracy = np.mean(predictedLabels == trueLabels) * 100
    return accuracy
