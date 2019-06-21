# mmc_control
Scripts used in academic papers about modular multilevel converters

## Matlab
The "matlab" folder contains matlab scripts to create the energy pulsation
data for the cases:
 - No compensation
 - analytical compensation
 - heuristic optimization

To create the data run "create_data.m".
Output is in the "matlab/data" folder.

## Python
The "python" folder contains the code for the CasADi based optimization.

Running "optimize_mmc.py" creates the results in the "python/data" folder.
