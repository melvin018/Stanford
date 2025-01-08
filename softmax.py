import download
from data_utils import load_CIFAR10
import data_utils
import numpy as np
from random import shuffle
from builtins import range



url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_data = "./data"
download.maybe_download_and_extract(url,CIFAR10_data)


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    cifar10_dir = './data/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir)
    try:
        del X_train, y_train
        del X_test, y_test
        print('Clear previously loaded data.')
    except:
        pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    mask_val = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask_val]
    y_val = y_train[mask_val]

    mask_train = list(range(num_training))
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    mask_test = list(range(num_test))
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    mask_dev = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask_dev]
    y_dev = y_train[mask_dev]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image   

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', y_dev.shape)



def softmax_loss_naive(W, X, y, reg):
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  for i in range(num_train): # type: ignore
      sco = np.dot(X[i], W)
      exp_sco = np.exp(sco)
      sum_sco = np.sum(exp_sco)
      loss -= np.log(exp_sco[y[i]] / sum_sco)
      dW += np.dot(X[i].T[:, np.newaxis], exp_sco[np.newaxis, :]) / sum_sco
      dW[:, y[i]] -= X[i].T
  loss = loss/X.shape[0] + reg*np.sum(W**2)
  dW = dW/X.shape[0] + 2*reg*W

  return loss, dW

import time

W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))


loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

from gradient_check import grad_check_sparse
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)


def softmax_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros_like(W)

  sco = np.dot(X, W)
  correct_sco = sco[range(X.shape[0]), y]
  exp_ = np.exp(sco - correct_sco[:, np.newaxis])
  log_ = np.log(exp_.sum(axis=1))
  loss = np.sum(log_) / X.shape[0]
  loss += reg * np.sum(W**2)

  dsco = exp_ / (exp_.sum(axis=1)[:, np.newaxis])
  dsco[range(X.shape[0]), y] -= 1
  dsco /= X.shape[0]
  dW = np.dot(X.T, dsco) + 2*reg*W

  return loss, dW


tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))


tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
print('Gradient difference: %f' % grad_difference)
