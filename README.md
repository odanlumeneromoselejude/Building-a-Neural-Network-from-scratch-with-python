# Building-a-Neural-Network-from-scratch-with-python
Here i will show you how to build/train a neural network to make predictions
activation_functions.py contains a list of all the basic activation functions for bot back propagation and front propaation
train.py contains the class and methods we need to train our network
test.py is where we actuall put the code into practice

# How to implement
1) Create an object of the train class and pass all the required parameters (training inputs, training outputs, learning rate, epoch, activation function and input so the neural network predicts its out put
train = train(input

train = train(inputs=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]),
              outputs=np.array([[1, 0, 0, 1, 1]]),
              lr=0.05,
              epoch=50000,
              activation="sigmoid",
              predict=np.array([1, 0, 1]))
train.train()
train.predict()

we set our training inputs to be a 5X3 matrix, learning rate of 0.1 and epoch to 5000
we want to predict the output of 1, 0, 1
This neural network can be improved by tuning the learning rate
a learning rate of 0.05 and epoch of 50000 will produce a better prediction with 99% accuracy
