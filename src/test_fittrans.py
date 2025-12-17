from sklearn.preprocessing import MinMaxScaler
import numpy as np

X = np.array([[1, 2, 3, 4],
              [1, 2, 3, 4],
              [1, 2, 3, 4],
              [3, 5, 7, 9],
              [1, 2, 3, 4]])
X_T = np.transpose(X)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_T_scaled = scaler.fit_transform(X_T)
X_T_scaled_T = np.transpose(X_T_scaled)

print('X:', X)
print('X_scaled:', X_scaled)
print('X_T:', X_T)
print('X_T_scaled:', X_T_scaled)
print('X_T_scaled_T:', X_T_scaled_T)

