import numpy as np
import matplotlib.pyplot as plt
#from preprocessing.polynomial_features import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

lr = LinearRegression()

#plt.plot(x,y,'o')
for i in range(1, 10):
    poly = PolynomialFeatures(degree=i)
    X = poly.transform(x)
    lr.fit_autograd(X,y, batch_size=5)
    y_hat = lr.predict(X)
    plt.plot(x, y_hat, label='degree = ' + str(i))
plt.legend()
plt.show()