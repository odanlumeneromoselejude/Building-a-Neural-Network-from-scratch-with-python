import numpy as np
from train import train

train = train(inputs=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]),
              outputs=np.array([[1, 0, 0, 1, 1]]),
              lr=0.05,
              epoch=500000,
              activation="sigmoid",
              predict=np.array([1, 0, 1]))
train.train()
train.predict()
