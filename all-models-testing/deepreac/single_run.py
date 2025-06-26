import numpy as np
from os import system
from sys import argv

seeds = np.genfromtxt('seeds.txt')
print(seeds)

system('rm -rf output.txt')

for j in [10, 25, 50, 75, 90]:
    i = int(argv[1])*2
    while i < (int(argv[1])*2 + 2):
        s = int(seeds[i])
        i += 1
        p = int(seeds[i])
        i += 1
        system(f'bash script.sh {s} {p} {j} {i // 2} {argv[2]}')
