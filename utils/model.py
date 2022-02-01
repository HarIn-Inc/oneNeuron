import numpy as np
import logging
from tqdm import tqdm
class perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4
    logging.info(f'Initial weights initialization:\n {self.weights}')
    self.eta = eta
    self.epochs = epochs

  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights)
    return np.where(z > 0, 1, 0)
  def fit(self, X, y):
    self.X = X
    self.y = y

    x_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
    logging.info(f'X with bias:\n {x_with_bias}')

    for epoch in tqdm(range(self.epochs), total=self.epochs, desc='Training the Model'):
      logging.info('--' * 10)
      logging.info(f'For Epoch:\n {epoch}')
      logging.info('--' * 10)

      y_hat = self.activationFunction(x_with_bias, self.weights) #Foreward Propogation
      logging.info(f'Predicted value after Foreward Pass:\n {y_hat}')
      self.error = self.y - y_hat
      logging.info(f'Error Value:\n {self.error}')

      self.weights = self.weights + self.eta * np.dot(x_with_bias.T, self.error) #Backward Propogation
      logging.info(f'Update weight after epoch:\n {epoch}/{self.epochs} :\n {self.weights}')
      logging.info('##' * 10)

  def predict(self, X):
    x_with_bias = np.c_[X, -np.ones((len(X), 1))]
    return self.activationFunction(x_with_bias, self.weights)

  def totalloss(self):
    total_loss = np.sum(self.error)
    logging.info(f'Total Loss:\n {total_loss}')
    return total_loss