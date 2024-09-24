# Hamiltonian Cycle Problem Solver
This solver was implemented for the SAT Solving Course at TU Wien.

## Implementation
The general idea behind this implementation is the reduction of the Hamiltonian Cycle Problem to the Hamiltonian Path Problem in polynomial time.
As a result, the problem that is actually encoded as SAT and then solved, is the Hamiltonian Path Problem.

## Usage
The solver uses `minisat`, which needs to be installed on the system. Furthermore, example graphs and solutions are given in the respective folders. For demonstration purposes of the reduction, graph0 is well suited.
To use the solver use the following command:
* `python3 main.py -g ./graphs/ -s ./solutions/ graph0`