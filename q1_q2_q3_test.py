
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_non_vectorised(X, y, batch_size=30)
    #LR.fit_autograd(X, y, batch_size=5)
    y_hat = LR.predict(X)
    plt.plot(y, y_hat, 'o')
    plt.show()
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
