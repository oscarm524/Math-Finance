import numpy as np

##################
## Sharpe Ratio ##
##################

def Sharpe_Ratio(w, mean_vec, sigma, rf):
   
   """
   This function computes the Sharpe Ratio
   w: vector of weights
   mean_vec: vector of sample means
   sigma: sample-covariance matrix
   rf: risk-free rate
   """
   
   ## Here we compute the numerator of Sharpe-Ratio
   num = np.dot(w, mean_vec) - rf
   
   ## Here we compute the denominator of the Sharpe-Ratio
   den = np.dot(np.transpose(w), np.inner(sigma, w))
   
   return num / np.sqrt(den)
