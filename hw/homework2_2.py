
import random
import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt

'''
The point here is to show a simpler model might do better out-of-sample.
Even though the complex model (linear) has a lower bias (see graph),
the variance turns out to be much higher compared to the simple model. 
'''


def bias_variance():
	number_trials = 10000
	x = np.linspace(-1,1,1000)
	y_hat_simple = np.zeros((number_trials,len(x)))
	y_hat_complex = np.zeros((number_trials,len(x)))
	#Not very Python-ic ! 
	x01 = np.zeros(number_trials)
	x11 = np.zeros(number_trials)
	x12 =  np.zeros(number_trials)
	x02 = np.zeros(number_trials)


	for i in range(number_trials):
		#Pick Two Data Point Randomly
		x_train = np.array(random.sample(x, 2))
		y_train = np.sin(pi* x_train)
		#Model One - Simple
		#Intercept
		x01[i] = np.mean(y_train)
		#Slope
		x11[i] = 0
		#Model Two - Linear
		#Slope
		x12[i] = (y_train[1] - y_train[0]) / (x_train[1] - x_train[0])
		#Intercept
		x02[i]=  y_train[0] - x12[i] * x_train[0]
		#Predictions
		y_hat_simple[i,:] = np.tile(x01[i],len(x)) 
		y_hat_complex[i,:] = x * x12[i] + x02[i]

	#Bias - Variance Calculation
	#Bias - Calculate the MSE for the average Function
	bias_simple = mse(np.sin(pi*x),np.mean(y_hat_simple,0))
	bias_complex = mse(np.sin(pi*x),np.mean(y_hat_complex,0))
	#Variance - Calculate the MSE for each realization and take the average. 
	variance_simple = np.mean([mse(bias_simple,var) for var in y_hat_simple])
	variance_complex = np.mean([mse(bias_complex,var) for var in y_hat_complex])

	print 'For simple model, the bias is {0} and the variance is {1}'.format(bias_simple,variance_simple)
	print 'For complex model, the bias is {0} and the variance is {1}'.format(bias_complex,variance_complex)

	print 'For simple model, the generalization error is {0}'.format(bias_simple+variance_simple)
	print 'For complex model, the generalization error is {0}'.format(bias_complex+variance_complex)


	#Bias Plot. Omitted the variance plot. 
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.plot(x,np.sin(x*pi),label='target')
	complex_bias = np.mean(y_hat_complex,0)
	ax.plot(x,complex_bias,'r',label='complex')
	simple_bias = np.mean(y_hat_simple,0) 
	ax.plot(x,simple_bias,'m',label='simple')
	ax.legend()
	plt.show()
	plt.savefig('Bias Chart')



def mse(y,h):
    return(np.mean(np.square(y-h)))

if __name__ == '__main__':
	bias_variance()



