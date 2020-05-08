#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# The function that applying Newton method to solve lambda for Lagrangian 
# # Parameters:
# sigma: an np.matrix representing covariance matrix
# w: an np.array representing the portfolio weights that need to be projected
# C: a number representing risk tolerance level
# return: lambda that solve the equation of Lagrangian method
def newtonLambda(sigma, w, C):
    identity = np.identity(len(w))
    # set 0 as the initial guess of lambda
    L = 0
    # compute f(lambda)
    fL = w.T.dot(np.linalg.inv(identity + L*sigma)).dot(sigma).dot(np.linalg.inv(identity + L*sigma)).dot(w) - C
    # iterate until f(lambda) is zero
    while (abs(fL) > 1e-12):
        # compute f'(lambda)
        fPrimeL = -2*(w.T.dot(np.linalg.inv(identity + L*sigma)).dot(sigma).dot(np.linalg.inv(identity + L*sigma)).dot(sigma).dot(np.linalg.inv(identity + L*sigma)).dot(w))
        # update lambda
        L -= fL/fPrimeL
        # update f(lambda)
        fL = w.T.dot(np.linalg.inv(identity + L*sigma)).dot(sigma).dot(np.linalg.inv(identity + L*sigma)).dot(w) - C
    return L

# The function that doing the projection
# # Parameters:
# sigma: an np.matrix representing covariance matrix
# w: an np.array representing the portfolio weights that need to be projected
# C: a number representing risk tolerance level
# return: the portfolio weights after projection
def projection(sigma, w, C):
    identity = np.identity(len(w))
    if (w.T.dot(sigma).dot(w) <= C):
        # we don't need to do projection
        return w
    else:
        # we need to do projection by Lagranian method
        # apply newton's method to find optimal lambda
        L = newtonLambda(sigma, w, C)
        return np.linalg.inv(identity + L*sigma).dot(w)

# The function that applying Descent method with gradient to find optimal portfolio
# Parameters:
# mu: an np.array representing expected return
# sigma: an np.matrix representing covariance matrix
# C: a number representing risk tolerance level
# wbm: benchmark index, default is an np.array with zeros that means no benchmark
# Return: the optimal portfolio that maximize return with constraints of variance
def optimalPortfolio(mu, sigma, C, wbm = np.zeros(2)):
    length = len(mu)
    # make 0 as the initial guess of w
    w1 = np.zeros(length)
    
    # gradient of this problem is mu
    # do the first update of w, w_2 = w_1 + d_1*mu
    # set parameters for Backtracking-Armijo Line Search
    alpha = 1.
    beta = .1
    # apply Backtracking-Armijo Line Search to find step length
    while (-(w1 + alpha*mu).dot(mu) >= -(w1.dot(mu) + alpha*beta*mu.dot(mu))):
        alpha /= 2.
    # do the projection of w_2 to get final w_2
    w2 = projection(sigma, w1 + alpha*mu, C)
    # iterate until optimal portfolio is reached
    # w_2 means w_(k+1), w_1 means w_k here
    while ((w2.dot(mu) - w1.dot(mu)) > 1e-8):
        # set w_k
        w1 = w2
        # set initial alpha for Backtracking-Armijo Line Search
        alpha = 1.
        # apply Backtracking-Armijo Line Search to find step length
        while (-(w1 + alpha*mu).dot(mu) >= -(w1.dot(mu) + alpha*beta*mu.dot(mu))):
            alpha /= 2.
        # do the projection of w_(k+1) to get final w_(k+1)
        w2 = projection(sigma, w1 + alpha*mu, C)
    # check if benchmark index is specified
    if ((wbm == 0).all()):
        # if no benchmark, directly return the result from previous part of this program
        return w2
    else:
        # if benchmark index exists, the previous part of this program is finding optimal w_hat
        # add benchmark index to optimal w_hat to get optimal portfolio
        return w2 + wbm


# In[2]:


# Test by giving expected return and covariance for the following asset classs of
# U.S. equities, foreign equities, long-dated U.S. bonds, and hedge funds.
# Benchmark index is zero

# guess of expected return
muGuess = np.array([.07, .08, .02, .05])
# guess of covariance matrix
sigmaGuess = np.array([[.16, .15, .02, .06], [.15, .24, .02, .08], [.02, .02, .03, .01], [.06, .08, .01, .10]])
# set different levels of risk tolerance
C = [5, 20, 50, 100]
for c in C:
    print("When the risk tolerance level is", c, ", the optimal portfolio weight is", optimalPortfolio(mu = muGuess, sigma = sigmaGuess, C = c))

