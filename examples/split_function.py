
import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(20).reshape((10, 2)), range(10)
print('X=', X)
print('y=', list(y))

shuffle_def=False
#shuffle_def=True
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, shuffle=shuffle_def)
print('shuffle=', shuffle_def)
print('X_train=', X_train)
print('y_train=', y_train)
print('X_test=', X_test)
print('y_test=', y_test)
