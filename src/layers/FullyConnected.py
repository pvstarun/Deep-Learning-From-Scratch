import numpy as np
from Layers.Base import BaseLayer # type: ignore

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)  # +1 for bias
        self._optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        bias = np.ones((batch_size, 1))
        input_with_bias = np.concatenate((input_tensor, bias), axis=1) # Adding bias term to the input tensor
        self.input_with_bias = input_with_bias
        return np.dot(input_with_bias, self.weights) # X.W

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(self.input_with_bias.T, error_tensor) #X.T . dL/dY
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        return np.dot(error_tensor, self.weights.T[:, :-1]) # dL/dX = dL/dY . W.T (excluding bias weights)

    @property
    def gradient_weights(self):
        return self._gradient_weights
