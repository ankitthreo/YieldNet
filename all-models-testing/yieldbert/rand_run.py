import numpy as np
from os import system
from sys import argv

seeds = np.genfromtxt('seeds.txt')
print(seeds)

for j in [0,1,2]:
    i = 0
    while i < len(seeds):
        s = int(seeds[i])
        i += 1
        p = int(seeds[i])
        i += 1
        system(f'bash run_one.sh {s} {p} {j} {i // 2}')
