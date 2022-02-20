import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

LR = LinearRegression()

start = time.time()
LR.fit_autograd(X, y, batch_size=5)
end = time.time()
print('Time taken for fit_autograd: ', end - start)

start = time.time()
LR.fit_normal(X, y)
end = time.time()
print('Time taken for fit_normal: ', end - start)