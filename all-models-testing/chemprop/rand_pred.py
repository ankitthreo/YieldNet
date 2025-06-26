import numpy as np
from os import system
from sys import argv

for j in [25, 40, 50, 60, 75]:
    i = 0
    while i < 10:
        i += 1
        system(f'bash predict_one.sh {j} {i} {argv[1]}')
