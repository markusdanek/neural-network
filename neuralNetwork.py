# Imports
import numpy as np

class BackPropagationNetwork:
    #
    # Class members
    #
    layerCount = 0
    shape = none
    weights = []

    #
    # Class methods
    #
    def __init__(self, layerSize):

        # Layer info
        self.layerCount = len(layerSize)
        self.shape = layerSize

        # Input/Output data from last run
        self._layerInput = []
        self._layerOutput = []

        # Create the weight arrays
        for (11,12) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale = 0.01, size = (12,11+1)))

    #
    # Run method
    #
    def Run(self, input):
        """ Run the network based on the input data """
        lnCases = input.shape[0]

        # Clear out the previous intermediate value lists
        self._layerInput = []
        self._layerOutput = []

        # Run it:
        for index in range(self.layerCount):
            # Determine layer input
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))
            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sgm(layerInput))
        return self._layerOutput[-1].T

    # Transfer function
    def sgm(self, x, Derivate = False):
        if not Derivate:
            return 1 / (1 + np.exp(-x))
        else:
            out = self.sgm(x)
            return out * (1 - out)

#
# If run as script, create a test object
#
if __name__ == "__main__":
    bpn = BackPropagationNetwork((2,2,1))
    print(bpn.shape)
    print(bpn.weights)

    lvInput = np.array([0,0], [1,1], [-1, 0.5])
    lvOutput = bpn.Run(lvInput)

    print("Input: {0}\nOutput: {1}".format(lvInput, lvOutput))
