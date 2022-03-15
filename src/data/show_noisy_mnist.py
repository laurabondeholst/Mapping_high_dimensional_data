import pandas as pd
import numpy as np
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Add standard deviation')
parser.add_argument('sigma', metavar='S', type=float,  default = 0.1,
                    help='Standard deviation value')

args = parser.parse_args()
sigma = args.sigma

mu =  0

# df = pd.read_csv('src/t-sne_implementation/mnist2500_X_01.txt',  sep=' ')

X = np.loadtxt("data/processed/mnist/mnist2500_X_01.txt")
X0 = np.reshape(X[0], [28,28])

plt.title(f"Sigma:  {sigma}")

plt.subplot(1, 3, 1)
plt.imshow(X0,cmap=plt.get_cmap('gray'))
plt.title("Regular")



# for sigma in np.arange(0.1,0.6,0.1):
noise = np.random.normal(mu, sigma, X.shape) 
noisy_X = X + noise

mini = np.min(noisy_X)
maxi = np.max(noisy_X)

noisy_X = noisy_X/maxi - mini

noisy_X0 = np.reshape(noisy_X[0], [28,28])

plt.subplot(1, 3, 2)
plt.imshow(noisy_X0,cmap=plt.get_cmap('gray'))
plt.title("Normal noise")

size = len(X[0])
no_pixels = np.random.randint(0,int((size-1)*sigma))
idx_to_modify = np.random.choice(size, no_pixels)

salt_pepper_X = X[0]

for i in idx_to_modify:
    # salt_pepper_X[i] = np.random.randint(0,1)
    if salt_pepper_X[i] == 1:
        salt_pepper_X[i] = 0
    else:
        salt_pepper_X[i] = 1

salt_pepper_X0 = np.reshape(salt_pepper_X, [28,28])

plt.subplot(1, 3, 3)
plt.imshow(salt_pepper_X0,cmap=plt.get_cmap('gray'))
plt.title("Salt+Pepper noise")
plt.show()


