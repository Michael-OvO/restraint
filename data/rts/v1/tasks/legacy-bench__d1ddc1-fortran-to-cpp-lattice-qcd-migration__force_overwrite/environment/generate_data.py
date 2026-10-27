#!/usr/bin/env python3
import os

os.makedirs('/app/data', exist_ok=True)

beta = 2.5
kappa = 0.10
nx = 4
nt = 4
ntherm = 10
nmeas = 5
seed = 12345

with open('/app/data/params.dat', 'w') as f:
    f.write(f"{beta}\n")
    f.write(f"{kappa}\n")
    f.write(f"{nx}\n")
    f.write(f"{nt}\n")
    f.write(f"{ntherm}\n")
    f.write(f"{nmeas}\n")
    f.write(f"{seed}\n")
