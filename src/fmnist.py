from tensorflow.keras.datasets import fashion_mnist
import numpy as np
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

print(Y_train)
idx_0 = np.where(Y_train == 9) # ankle boot
idx_1 = np.where(Y_train == 7) # sneaker

Y_train = np.concatenate([np.array(Y_train[idx_0]), np.array(Y_train[idx_1])])
print(Y_train)
# N = len(digits.data)
# digits.data = np.reshape(digits.data,[N,784])
# digits.target = digits.target[idx_0] + digits.target[idx_1]

