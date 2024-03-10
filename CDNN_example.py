import cdeeply_neural_network
from numpy import random, array, mat, sin, pi

numFeatures = 10
numSamples = 100
noiseAmplitude = 0.1

random.seed()
NNtypes = [ "autoencoder with 1 latent feature", "regressor" ]


    # generate a training matrix that traces out some noisy curve in Nf-dimensional space (noise ~ 0.1)

print("Training data along a 1D curve in feature space")
print("  * %i samples, %i features; feature variance ~1 + Gaussian noise ~%g" % (numSamples, numFeatures, noiseAmplitude))
dependentVar = random.rand(numSamples+1, 1)
trainTestMat = mat(noiseAmplitude*random.randn((numSamples+1)*numFeatures)).reshape(numSamples+1, numFeatures)
for cf in range(numFeatures):
    featurePhase = 2*pi*random.rand(1)
    featureCurvature = 2*pi*random.rand(1)
    trainTestMat[:, cf] = trainTestMat[:, cf] + sin(featureCurvature*dependentVar + featurePhase)

NN = cdeeply_neural_network.CDNN()

for c2 in range(2):
    
    
        # train a neural network from our matrix
    
    print("Generating " + NNtypes[c2])
    if c2 == 0:
        outputsComputedByServer = NN.tabular_encoder(trainTestMat[range(numSamples)], "SAMPLE_FEATURE_ARRAY",
            numEncodingFeatures=1, doEncoder=True, doDecoder=True)
        firstSampleOutputs = array(NN.runSample(trainTestMat[0]))       # make a copy; otherwise will be overwritten by the next function call
        testSampleOutputs = NN.runSample(trainTestMat[numSamples])
    else:
        outputsComputedByServer = NN.tabular_regressor(trainTestMat[range(numSamples)], "SAMPLE_FEATURE_ARRAY", [numFeatures])
        firstSampleOutputs = array(NN.runSample(trainTestMat[0, range(numFeatures-1)]))
        testSampleOutputs = NN.runSample(trainTestMat[numSamples, range(numFeatures-1)])
    
    if max(abs(firstSampleOutputs[0, :]-outputsComputedByServer[0, :])) > .0001:         # sanity check using output 1
        raise ValueError(["  ** Network problem?  Sample 1 output was calculated as " + str(firstSampleOutputs[0, :]) \
            + " locally vs " + str(outputsComputedByServer[0, :]) + " by the server"])
    
    
        # run the network on the test sample
    
    if c2 == 0:
        targetValue = trainTestMat[-1, 0]
        targetDescription = "reconstructed feature 1"
    else:
        targetValue = trainTestMat[-1, -1]
        targetDescription = "output"
    print("  Test sample:  " + targetDescription + " was %g; target value was %g" % (testSampleOutputs[0, 0], targetValue))
