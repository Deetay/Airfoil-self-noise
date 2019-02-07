import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sb

plt.ion();

data = np.loadtxt('airfoil_self_noise.dat', usecols=(0, 1, 2, 3, 4))
target = np.loadtxt('airfoil_self_noise.dat', usecols=(5))

sb.pairplot(pd.DataFrame(data), diag_kind="kde")
plt.show()

data = StandardScaler().fit_transform(data)
data_train, data_test, target_train, target_test = \
    train_test_split(data, target, test_size=0.2, random_state=17)

linear_regression = LinearRegression()
linear_regression.fit(data_train, target_train)
test_pred = linear_regression.predict(data_test)
data_pred = linear_regression.predict(data)

print("Test data prediction score: %.2f" % r2_score(target_test, test_pred))
print("Mean squared error: %.2f" % mean_squared_error(target_test, test_pred))
#
# x = np.linspace(100, 140, 1000)
# tmp = np.polyfit(target_test,test_pred, 1)
# a=tmp[0]
# b=tmp[1]
# plt.plot(x,x)
# plt.plot(x,a*x+b)
# plt.xlabel('Measured sound pressure level')
# plt.ylabel('Predicted sound pressure level')
# plt.plot(target_test, test_pred, 'ro')
# plt.show()
#
# plt.plot(data_test,target_test,'b.', data_test,test_pred,'r.')
# plt.show()
