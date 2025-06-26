import numpy as np

N = int(2 ** 10)
seeds = np.random.choice(N,20)
np.savetxt('seeds.txt',seeds)
