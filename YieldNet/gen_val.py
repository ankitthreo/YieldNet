import random
import numpy as np

seeds = np.genfromtxt('seeds.txt')
for j in range(len(seeds)):
    if j % 2 == 0:
        random.seed(int(seeds[j]))
        for i in range(100):
            perm = list(range(1000))
            random.shuffle(perm)
            np.savetxt(f"val_perms/{j // 2 + 1}/epoch_{i}.txt", perm)
