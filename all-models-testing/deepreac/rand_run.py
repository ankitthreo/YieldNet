import numpy as np
from os import system
from sys import argv

seeds = np.genfromtxt('seeds.txt')
print(seeds)

system('rm -rf output.txt')

for j in [25, 40, 50, 60, 75]:
    i = 0
    while i < len(seeds):
        s = int(seeds[i])
        i += 1
        p = int(seeds[i])
        i += 1
        system(f'bash script.sh {s} {p} {j} {i // 2} {argv[1]}')
