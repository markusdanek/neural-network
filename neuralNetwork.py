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
            self.weights.append(np.random.normal(scale=0.01, size=(12,11+1)))

#
# If run as script, create a test object
#
if __name__ == "__main__":
    bpn = BackPropagationNetwork((2,2,1))
    print(bpn.shape)
    print(bpn.weights)
