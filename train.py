import numpy as np
import activation_functions as af


class train:
    def __init__(self, inputs, outputs, lr, epoch, activation, predict):
        # lr is the learning rate. Learning rate increases the efficiency of neural network
        self.inputs = inputs
        self.outputs = outputs
        self.lr = lr
        self.epoch = epoch
        self.activation = activation
        self.prediction = predict
        self.weights = ""
        self.bias = ""
        self.x = ""
        self.layers = False

    def fprop(self):
        if not self.layers:
            input_row, input_column = self.inputs.shape
            outputs_row, output_column = self.outputs.shape

            self.outputs = self.outputs.reshape(output_column, outputs_row)
            self.weights = np.random.rand(input_column, 1)
            self.bias = np.random.rand(1)
            self.layers = True

        self.x = np.dot(self.inputs, self.weights) + self.bias
        if self.activation == "sigmoid":
            return af.sigmoid(self.x, False)
        elif self.activation == "tanh":
            return af.tanh(self.x, False)
        elif self.activation == "relu":
            return af.relu(self.x)
        else:
            return af.sigmoid(self.x, False)

    def bprop(self):
        for epoch in range(self.epoch):
            dl_dp = self.fprop() - self.outputs
            if self.activation == "sigmoid":
                dp_dc = af.sigmoid(self.x, True)
            elif self.activation == "tanh":
                dp_dc = af.tanh(self.x, True)
            elif self.activation == "relu":
                dp_dc = af.relu(self.x)
            else:
                dp_dc = af.sigmoid(self.x, True)
            dc_dw = self.inputs.T

            dl_dw = np.dot(dc_dw, dl_dp*dp_dc)
            self.weights = self.weights - (self.lr * dl_dw)

            for n in (dl_dp*dp_dc):
                self.bias = self.bias - (self.lr * n)
        return self.weights,self.bias

    def train(self):
        return self.bprop()

    def predict(self):
        weight, bias = self.bprop()
        x = np.dot(self.prediction, weight) + bias
        print("\nPredicted result:")
        if self.activation == "sigmoid":
            print(af.sigmoid(x, False))
            if af.sigmoid(x, False) <= 0.5:
                accuracy = str(int(100 - (abs((af.sigmoid(x, False)[0] - 0)) * 100)))+"%"
                print("Accuracy:" + " " + accuracy)
            else:
                accuracy = str(int(100 - (abs((af.sigmoid(x, False)[0] - 1)) * 100)))+"%"
                print("Accuracy:" + " " + accuracy)
        elif self.activation == "tanh":
            print(af.tanh(x, False))
        elif self.activation == "relu":
            print(af.relu(x))
        else:
            print(af.sigmoid(x, False))

