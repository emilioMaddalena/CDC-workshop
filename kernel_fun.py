'''
Author: Emilio Maddalena
Date: Nov 2022
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary, set_trainable

import params

np.random.seed(0)

N     = 80  # number of samples
sigma = 0.2 # noise variance

# Sampling the ground-truth 
xmin  = -3*np.pi
xmax  =  3*np.pi

delta = 2
X1 = np.random.uniform(xmin, -delta, (int(N/2), 1))
X2 = np.random.uniform(delta, xmax, (int(N/2), 1))
X  = np.concatenate([X1,X2], axis=0)

delta = np.random.normal(0, sigma, (N, 1))
Y     = np.sin(X) + 0.015*np.square(X) + delta 

# Dataset needs to be converted to tensor for GPflow to handle it
data  = (tf.convert_to_tensor(X, "float64"), tf.convert_to_tensor(Y, "float64"))

# Defining the GP
kernel = gpflow.kernels.SquaredExponential()
my_gp  = gpflow.models.GPR(data, kernel=kernel) 

# Let's take a look at its hyperparameters (before training)
print_summary(my_gp)

# Picking an optimizer and training the GP through MLE
opt = gpflow.optimizers.Scipy()
opt.minimize(my_gp.training_loss, my_gp.trainable_variables, tol=1e-11, options=dict(maxiter=1000), method='l-bfgs-b')

# Let's take a look at its hyperparameters (after training)
print_summary(my_gp)

# Gridding the space and predicting!
xx = np.linspace(xmin * 1.15, xmax * 1.15, 1000).reshape(-1, 1) 
mean, var = my_gp.predict_f(xx)

# Plotting the results (two standard deviations = 95% confidence)
font_size = 10
fig = plt.figure()
fig.set_size_inches(10, 6)
plt.rcParams.update({'font.family':'Helvetica'})
plt.xlabel('$x$', fontsize=font_size)
plt.ylabel('$f(x)$', fontsize=font_size)

plt.plot(xx, mean, color=params.colors["red"], lw=2)
plt.fill_between(xx[:,0],
                 mean[:,0] - 2*np.sqrt(var[:,0]),
                 mean[:,0] + 2*np.sqrt(var[:,0]),
                 color=params.colors["red"],
                 alpha=0.2)
plt.plot(X, Y, "o", color=params.colors["grey"], ms=3.5)
#plt.xlabel('x'), plt.ylabel('f(x)')
#plt.title('Gaussian process regression with ' + str(N) + ' samples')

plt.xlim(xmin,xmax)
plt.ylim(-2,3)
fig.tight_layout()
plt.show()

filename = 'ex1.pdf'
fig.savefig(filename, bbox_inches='tight')
