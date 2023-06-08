import numpy as np

class FC:
    def __init__(self, input_size : int, output_size : int, name : str, initialize_method : str="random"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.initialize_method = initialize_method
        self.parameters = [self.initialize_weights(), self.initialize_bias()]
        self.input_shape = None
        self.reshaped_shape = None
    
    def initialize_weights(self):
        if self.initialize_method == "random":
            # TODO: Initialize weights with random values using np.random.rand
            return np.random.rand(self.output_size, self.input_size) * 0.01

        elif self.initialize_method == "xavier":
            return None

        elif self.initialize_method == "he":
            return None
        else:
            raise ValueError("Invalid initialization method")
    
    def initialize_bias(self):
        # TODO: Initialize bias with zeros
        return np.zeros((self.output_size, 1))
    
    def forward(self, A_prev):
        """
        Forward pass for fully connected layer.
            args:
                A_prev: activations from previous layer (or input data)
                A_prev.shape = (batch_size, input_size)
            returns:
                Z: output of the fully connected layer
        """
        # NOTICE: BATCH_SIZE is the first dimension of A_prev
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)

        # TODO: Implement forward pass for fully connected layer
        if len(self.input_shape) == 4:
            batch_size = self.input_shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T
        self.reshaped_shape = A_prev_tmp.shape
        
        # TODO: Forward part
        W, b = self.parameters
        Z = W @ A_prev_tmp + b
        return Z
    
    def backward(self, dZ, A_prev):
        """
        Backward pass for fully connected layer.
            args:
                dZ: derivative of the cost with respect to the output of the current layer
                A_prev: activations from previous layer (or input data)
            returns:
                dA_prev: derivative of the cost with respect to the activation of the previous layer
                grads: list of gradients for the weights and bias
        """
        A_prev_tmp = np.copy(A_prev)
        if len(A_prev_tmp.shape) == 4:
            batch_size = A_prev_tmp.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T

        # TODO: backward part
        W, b = self.parameters
        dW = (dZ @ A_prev_tmp.T) / A_prev.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / A_prev.shape[1]
        dA_prev = W.T @ dZ
        grads = [dW, db]
        if len(self.input_shape) == 4:
            dA_prev = dA_prev.T.reshape(self.input_shape)
        return dA_prev, grads
    
    def update(self, optimizer, grads):
        """
        Update the parameters of the layer.
            args:
                optimizer: optimizer object
                grads: list of gradients for the weights and bias
        """
        self.parameters = optimizer.update(grads, self.name)