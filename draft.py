import numpy as np

mask = np.random.randn(9,9)
print(mask.shape)
ratio = np.multiply(*mask.shape) / np.sum(mask, axis=None)
print(type(ratio))