import numpy as np

DATA_PATH = "data/processed/mnist/mnist2500_X_01.txt"
DATA_OUTPUT_PATH = "data/processed/noisy_mnist/"

X = np.loadtxt(DATA_PATH)

mu, sigma = 0, 5.0

size = len(X[0])
# for sigma in np.arange(0.1,1.1,0.1):
print(f"Creating noisy dataset for mu: {mu} and sigma {sigma}")

noise = np.random.normal(mu, sigma, X.shape) 
noisy_X = X + noise

mini = np.min(noisy_X)
maxi = np.max(noisy_X)

noisy_X = noisy_X/maxi - mini

# no_pixels = np.random.randint(0,int((size-1)*sigma))
# idx_to_modify = np.random.choice(size, no_pixels)

# noisy_X = X

# for i in range(len(noisy_X)):
#     for index in idx_to_modify:
#         # salt_pepper_X[i] = np.random.randint(0,1)
#         if noisy_X[i][index] == 1:
#             noisy_X[i][index] = 0
#         else:
#             noisy_X[i][index] = 1

np.savetxt(DATA_OUTPUT_PATH + f"mnist2500_X_01_sigma{int(sigma*10)}.txt",noisy_X)





