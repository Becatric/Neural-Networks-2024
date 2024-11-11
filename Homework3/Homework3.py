import numpy as np
from dataFunct import downloadMnist,encode,createBatches
from training import trainModel, evaluateModel


numInputs = 784
numHidden = 100
numClasses = 10
batchSize = 100
np.random.seed(0)

weightsHidden = np.random.randn(numInputs, numHidden) * 0.01
biasHidden = np.zeros((1, numHidden))
weightsOutput = np.random.randn(numHidden, numClasses) * 0.01
biasOutput = np.zeros((1, numClasses))
learningRate = 0.01
epochs = 50
dropoutRate = 0.2

trainFeatures, trainLabels = downloadMnist(True)
testFeatures, testLabels = downloadMnist(False)
trainFeatures = trainFeatures / 255.0
testFeatures = testFeatures / 255.0

trainLabelsEncoded = encode(trainLabels.flatten(), numClasses)
testLabelsEncoded = encode(testLabels.flatten(), numClasses)

trainBatches = createBatches(trainFeatures, trainLabelsEncoded, batchSize)

trainedWeightsHidden, trainedBiasHidden, trainedWeightsOutput, trainedBiasOutput = trainModel(
    trainBatches, weightsHidden, biasHidden, weightsOutput, biasOutput, learningRate, epochs, dropoutRate
)

testAccuracy = evaluateModel(testFeatures, testLabelsEncoded, trainedWeightsHidden, trainedBiasHidden, trainedWeightsOutput, trainedBiasOutput)
print(f"Test accuracy: {testAccuracy:.2f}%")
