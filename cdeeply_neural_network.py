# usage:
# 
# 0) Create a class instance:
# 
# myNN = CDNN()
# 
# 
# 1) Generate a neural network, using either of:
# 
# myNN.tabular_regressor( trainingSamples, indexOrder, outputIndices, importances,
#               maxWeights, maxHiddenNeurons, maxLayers, maxWeightDepth, maxActivationRate,
#               maxWeightsHardLimit, maxHiddenNeuronsHardLimit, maxActivationsHardLimit, allowedAFs[5], 
#               ifQuantizeW, wQuantBits, wQuantZeroInt, wQuantRange,
#               ifQuantizeY, yQuantBits, yQuantZeroInt, yQuantRange,
#               sparseWeights, allowNegativeWeights, hasBias, allowIOconnections )
# 
# myNN.tabular_encoder( trainingSamples, indexOrder, importances,
#               doEncoder, doDecoder, numEncodingFeatures, numVariationalFeatures, variationalDistribution,
#               maxWeights, maxHiddenNeurons, maxLayers, maxWeightDepth, maxActivationRate,
#               maxWeightsHardLimit, maxHiddenNeuronsHardLimit, maxActivationsHardLimit, allowedAFs[5],
#               ifQuantizeW, wQuantBits, wQuantZeroInt, wQuantRange,
#               ifQuantizeY, yQuantBits, yQuantZeroInt, yQuantRange,
#               sparseWeights, allowNegativeWeights, hasBias )
# 
# * indexOrder="SAMPLE_FEATURE_ARRAY" for trainingSamples[sampleNo][featureNo] indexing,
#       or "FEATURE_SAMPLE_ARRAY" for trainingSamples[featureNo][sampleNo] indexing.
# * For supervised x->y regression, the sample table contains BOTH 'x' and 'y', the latter specified by outputIndices[].
# * The importances table, if not empty, has dimensions numOutputFeatures and numSamples (ordered by indexOrder),
#       and weights the training cost function:  C = sum(Imp*dy^2).
# * Weight/neuron/etc limits are either positive integers or "NO_MAX".
# * variationalDistribution is either "UNIFORM_DIST" ([0, 1]) or "NORMAL_DIST" (mean=0, variance=1).
# * doEncoder, doDecoder, hasBias, and allowIOconnections are all Booleans.
# Both functions return the network outputs from the training data, if you care to check that it agrees with what's computed locally.
# 
# 
# 2) Run the network on a (single) new sample
# 
# oneSampleOutput = myNN.runSample(oneSampleInput [, oneSampleVariationalInput])
# 
# where oneSampleInput is a list of length numInputFeatures, and oneSampleOutput is a list of length numOutputFeatures.
# * If it's an autoencoder (encoder+decoder), length(oneSampleInput) and length(oneSampleOutput) equal the size of the training sample space.
#       If it's just an encoder, length(oneSampleOutput) equals numEncodingFeatures; if decoder only, length(oneSampleInput) must equal numEncodingFeatures.
# * If it's a decoder or autoencoder network having numVariationalFeatures > 0, then oneSampleVariationalInput is a list
#       of length numVariationalFeatures containing random numbers drawn from variationalDistribution.


import numpy as np
try:
    from urllib import parse, request
except:
    import urllib
    import urllib2


class CDNN:
    
    numLayers = encoderLayer = variationalLayer = -1
    layerSize = layerAFs = layerInputs = wSize = n0 = nf = weights = []
    y = []
    sparseWeights = False
    
    def identity(x):
        return x
    
    def step(x):
        return max(0, min(1, np.ceil(x)))
    
    def ReLU(x):
        return max(0, x)
    
    def ReLU1(x):
        return max(0, min(1, x))
    
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))
    
    fs = [ identity, step, ReLU, ReLU1, sigmoid, np.tanh ]
    
    def tabular_regressor(self,
            trainingSamples, indexOrder, outputIndices, importances=[],
            maxWeights="NO_MAX", maxHiddenNeurons="NO_MAX", maxLayers="NO_MAX", maxWeightDepth="NO_MAX", maxActivationRate=1.,
            maxWeightsHardLimit=True, maxHiddenNeuronsHardLimit=True, maxActivationsHardLimit=True, allowedAFs=[True,True,True,True,True],
            ifQuantizeW=False, wQuantBits=0, wQuantZeroInt=0, wQuantRange=1.,
            ifQuantizeY=False, yQuantBits=0, yQuantZeroInt=0, yQuantRange=1.,
            sparseWeights=False, allowNegativeWeights=True, hasBias=True, allowIOconnections=True):
        
        [ numFeatures, numSamples ] = self.getDims(trainingSamples.shape, indexOrder)
        numOutputs = len(outputIndices)
        numInputs = numFeatures - numOutputs
        self.sparseWeights = sparseWeights
        
        [ sampleString, rowcolString ] = self.CDNN_data2table(trainingSamples, numInputs+numOutputs, numSamples, indexOrder)
        if len(importances) > 0:
            importancesString = self.CDNN_data2table(importances, numInputs, numSamples, indexOrder)
        else:
            importancesString = ""
        
        orcStrings = []
        for rc in range(len(outputIndices)):
            orcStrings.append(str(outputIndices[rc]))
        outputRowsColumnsString = ",".join(orcStrings)
        
        formDict = {
            "samples": sampleString,
            "importances": importancesString,
            "rowscols": rowcolString,
            "rowcolRange": outputRowsColumnsString,
            "maxWeights": self.maxString(maxWeights),
            "maxNeurons": self.maxString(maxHiddenNeurons),
            "maxLayers": self.maxString(maxLayers),
            "maxWeightDepth": self.maxString(maxWeightDepth),
            "maxActivationRate": str(maxActivationRate),
            "maxWeightsHardLimit": self.ifChecked(maxWeightsHardLimit),
            "maxNeuronsHardLimit": self.ifChecked(maxHiddenNeuronsHardLimit),
            "maxActivationsHardLimit": self.ifChecked(maxActivationsHardLimit),
            "step": self.ifChecked(allowedAFs[0]),
            "ReLU": self.ifChecked(allowedAFs[1]),
            "ReLU1": self.ifChecked(allowedAFs[2]),
            "sigmoid": self.ifChecked(allowedAFs[3]),
            "tanh": self.ifChecked(allowedAFs[4]),
            "quantizeWeights": self.ifChecked(ifQuantizeW),
            "wQuantBits": str(wQuantBits),
            "wQuantZero": str(wQuantZeroInt),
            "wQuantRange": str(wQuantRange),
            "quantizeActivations": self.ifChecked(ifQuantizeY),
            "yQuantBits": str(yQuantBits),
            "yQuantZero": str(yQuantZeroInt),
            "yQuantRange": str(yQuantRange),
            "sparseWeights": self.ifChecked(sparseWeights),
            "allowNegativeWeights": self.ifChecked(allowNegativeWeights),
            "hasBias": self.ifChecked(hasBias),
            "allowIO": self.ifChecked(allowIOconnections),
            "submitStatus": "Submit",
            "NNtype": "regressor",
            "formSource": "Python_API"
        }
        
        return self.buildCDNN(formDict, numOutputs, numSamples, indexOrder)
    
    
    def tabular_encoder(self,
            trainingSamples, indexOrder, importances=[],
            doEncoder=True, doDecoder=True, numEncodingFeatures=1, numVariationalFeatures=0, variationalDistribution="NORMAL_DIST",
            maxWeights="NO_MAX", maxHiddenNeurons="NO_MAX", maxLayers="NO_MAX", maxWeightDepth="NO_MAX", maxActivationRate=1.,
            maxWeightsHardLimit=True, maxHiddenNeuronsHardLimit=True, maxActivationsHardLimit=True, allowedAFs=[True,True,True,True,True],
            ifQuantizeW=False, wQuantBits=0, wQuantZeroInt=0, wQuantRange=1.,
            ifQuantizeY=False, yQuantBits=0, yQuantZeroInt=0, yQuantRange=1.,
            sparseWeights=False, allowNegativeWeights=True, hasBias=True):
        
        [ numFeatures, numSamples ] = self.getDims(trainingSamples.shape, indexOrder)
        self.sparseWeights = sparseWeights
        
        [ sampleString, rowcolString ] = self.CDNN_data2table(trainingSamples, numFeatures, numSamples, indexOrder)
        if len(importances) > 0:
            importancesString = self.CDNN_data2table(importances, numFeatures, numSamples, indexOrder)
        else:
            importancesString = ""
        
        if variationalDistribution == "UNIFORM_DIST":
            variationalDistStr = "uniform"
        elif variationalDistribution == "NORMAL_DIST":
            variationalDistStr = "normal"
        else:
            raise ValueError("Variational distribution must be either \"UNIFORM_DIST\" or \"NORMAL_DIST\"")
        
        formDict = {
            "samples": sampleString,
            "importances": importancesString,
            "rowscols": rowcolString,
            "numFeatures": str(numEncodingFeatures),
            "doEncoder": self.ifChecked(doEncoder),
            "doDecoder": self.ifChecked(doDecoder),
            "numVPs": str(numVariationalFeatures),
            "variationalDist": variationalDistStr,
            "maxWeights": self.maxString(maxWeights),
            "maxNeurons": self.maxString(maxHiddenNeurons),
            "maxLayers": self.maxString(maxLayers),
            "maxWeightDepth": self.maxString(maxWeightDepth),
            "maxActivationRate": str(maxActivationRate),
            "maxWeightsHardLimit": self.ifChecked(maxWeightsHardLimit),
            "maxNeuronsHardLimit": self.ifChecked(maxHiddenNeuronsHardLimit),
            "maxActivationsHardLimit": self.ifChecked(maxActivationsHardLimit),
            "step": self.ifChecked(allowedAFs[0]),
            "ReLU": self.ifChecked(allowedAFs[1]),
            "ReLU1": self.ifChecked(allowedAFs[2]),
            "sigmoid": self.ifChecked(allowedAFs[3]),
            "tanh": self.ifChecked(allowedAFs[4]),
            "quantizeWeights": self.ifChecked(ifQuantizeW),
            "wQuantBits": str(wQuantBits),
            "wQuantZero": str(wQuantZeroInt),
            "wQuantRange": str(wQuantRange),
            "quantizeActivations": self.ifChecked(ifQuantizeY),
            "yQuantBits": str(yQuantBits),
            "yQuantZero": str(yQuantZeroInt),
            "yQuantRange": str(yQuantRange),
            "sparseWeights": self.ifChecked(sparseWeights),
            "allowNegativeWeights": self.ifChecked(allowNegativeWeights),
            "hasBias": self.ifChecked(hasBias),
            "submitStatus": "Submit",
            "NNtype": "autoencoder",
            "formSource": "Python_API"
        }
        
        if doDecoder:
            numOutputs = numFeatures
        else:
            numOutputs = numEncodingFeatures
        
        return self.buildCDNN(formDict, numOutputs, numSamples, indexOrder)
    
    
    def runSample(self, sampleInput, sampleVariationalInput=[]):
        
        self.y[0][0,0] = 1
        self.y[1] = sampleInput
        if self.variationalLayer >= 0:
            self.y[self.variationalLayer] = sampleVariationalInput
        
        for l in range (2, self.numLayers):
            if l != self.variationalLayer:
                self.y[l][:] = 0.
                for li in range(len(self.layerInputs[l])):
                    l0 = self.layerInputs[l][li]
                    if self.sparseWeights:
                        for w in range(self.wSize[l][li]):
                            self.y[l][0,self.nf[l][li][w]] = self.y[l][0,self.nf[l][li][w]] + self.weights[l][li][w] * self.y[l0][0,self.n0[l][li][w]]
                    else:
                        self.y[l] = self.y[l] + self.y[l0] * self.weights[l][li]
                for n in range(self.layerSize[l]):
                    self.y[l][0,n] = self.fs[self.layerAFs[l]](self.y[l][0,n])
        
        return self.y[-1]
    
    
    def ifChecked(self, checkedBool):
        if checkedBool:
            return "on"
        else:
            return ""
    
    def maxString(self, maxVar):
        if maxVar == "NO_MAX":
            return ""
        else:
            return str(maxVar)
    
    
    def getDims(self, sampleTableSize, transpose):
        
        if transpose == "FEATURE_SAMPLE_ARRAY":
            return  [ sampleTableSize[0], sampleTableSize[1] ]
        elif transpose == "SAMPLE_FEATURE_ARRAY":
            return  [ sampleTableSize[1], sampleTableSize[0] ]
        else:
            raise ValueError("transpose must be either \"FEATURE_SAMPLE_ARRAY\" or \"SAMPLE_FEATURE_ARRAY\"")
    
    
    def CDNN_data2table(self, data, numIOs, numSamples, transpose):
        
        rowcolStrings = [ "rows", "columns" ]
        
        if transpose == "FEATURE_SAMPLE_ARRAY":
            dim1 = numIOs
            dim2 = numSamples
            rowcol = "rows"
        elif transpose == "SAMPLE_FEATURE_ARRAY":
            dim1 = numSamples
            dim2 = numIOs
            rowcol = "columns"
        
        tableRowStrings = []
        for i1 in range(dim1):
            rowElStrings = []
            for i2 in range(dim2):
                rowElStrings.append(str(data[i1, i2]))
            tableRowStrings.append(",".join(rowElStrings))
        
        tableStr = "\n".join(tableRowStrings)
        
        return [ tableStr, rowcol ]
    
    
    def buildCDNN(self, formDict, numOutputs, numSamples, transpose):
        
        def loadNumArray(numericString):
            numStrings = numericString.split(",")
            numStrings[:] = [s for s in numStrings if len(s) > 0]
            numArray = []
            for n in range(len(numStrings)):
                numArray.append(float(numStrings[n]))
            return numArray
        
        
        try:
            formData = parse.urlencode(formDict).encode()
            NNdata = request.urlopen(request.Request("https://cdeeply.com/myNN.php", data=formData)).read()
        except:
            formData = urllib.urlencode(formDict).encode()
            NNdata = urllib2.urlopen(urllib2.Request("https://cdeeply.com/myNN.php", data=formData)).read()
        
        firstChar = ord(NNdata[0])
        if firstChar < ord('0') or firstChar > ord('9'):
            raise ValueError(NNdata)
        
        NNdataRows = NNdata.split(";")
        
        NNheader = NNdataRows[0].split(",")
        self.numLayers = int(NNheader[0])
        self.encoderLayer = int(NNheader[1])-1
        self.variationalLayer = int(NNheader[2])-1
        
        self.layerSize = [ int(f) for f in loadNumArray(NNdataRows[1]) ]
        self.layerAFs = [ int(f) for f in loadNumArray(NNdataRows[2]) ]
        numLayerInputs = [ int(f) for f in loadNumArray(NNdataRows[3]) ]
        
        allLayerInputs = [ int(f) for f in loadNumArray(NNdataRows[4]) ]
        self.layerInputs = []
        self.wSize = []
        self.n0 = []
        self.nf = []
        if self.sparseWeights:
            allWsizes = [ int(f) for f in loadNumArray(NNdataRows[5]) ]
            allN0s = [ int(f) for f in loadNumArray(NNdataRows[6]) ]
            allNfs = [ int(f) for f in loadNumArray(NNdataRows[7]) ]
            swOffset = 3
        else:
            swOffset = 0
        
        idx = 0
        for l in range(self.numLayers):
            self.layerInputs.append(allLayerInputs[idx:idx+numLayerInputs[l]])
            if self.sparseWeights:
                self.wSize.append(allWsizes[idx:idx+numLayerInputs[l]])
            idx += numLayerInputs[l]
        
        allWs = loadNumArray(NNdataRows[5+swOffset])
        self.weights = []
        idx = 0
        for l in range(self.numLayers):
            if self.sparseWeights:
                self.n0.append([])
                self.nf.append([])
            self.weights.append([])
            for li in range(numLayerInputs[l]):
                l0 = self.layerInputs[l][li]
                if self.sparseWeights:
                    numWeights = self.wSize[l][li]
                    self.n0[l].append(allN0s[idx:idx+numWeights])
                    self.nf[l].append(allNfs[idx:idx+numWeights])
                    self.weights[l].append(allWs[idx:idx+numWeights])
                else:
                    numWeights = self.layerSize[l0]*self.layerSize[l]
                    self.weights[l].append(np.mat(allWs[idx:idx+numWeights]).reshape((self.layerSize[l0], self.layerSize[l]), order='F'))
                idx += numWeights
        
        sampleOutputs = np.array(loadNumArray(NNdataRows[6+swOffset])).reshape((numOutputs, numSamples), order='C');
        if transpose == "SAMPLE_FEATURE_ARRAY":
            sampleOutputs = np.transpose(sampleOutputs)
        
        self.y = []
        for l in range(self.numLayers):
            self.y.append(np.mat([0.] * self.layerSize[l]).reshape((1, self.layerSize[l])))
        
        return sampleOutputs
