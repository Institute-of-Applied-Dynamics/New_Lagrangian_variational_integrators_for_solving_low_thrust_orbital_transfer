# Code for the paper Variational integrators for a new Lagrangian approach to optimal control affine systems with a quadratic Lagrange term

This code can be used to generate the low-thrust orbitral transfer optimal control problem treated as an example in the paper 'Variational integrators for a new Lagrangian approach to optimal control affine systems with a quadratic Lagrange term' by Konopik et al, using variational integrators derved from a new Lagrangian approach to solving optimal control problems.

It consists of two parts:
1. A program that generates the data sets, Generate_solutions.ipynb
2. The script Data_analysis.ipynb that generates the figures of the paper once all data sets are generated


## The data sets needed for generating the data:

1. $h=0.1$, $\alpha=\beta=\gamma=1$ both control dependent and independent approaches
2. $h=0.1$, $\alpha=\beta=\gamma=0.5$ both control dependent and independent approaches
3. Standard approach solution with $h=0.1$
4. $h=0.01$, $\alpha=\beta=\gamma=0.5$  control independent approach
5. $h=0.1,0.4,0.8$ $\alpha=\beta=\gamma=1$  control independent approach
6. $h=0.1,0.4,0.8$ $\alpha=1,\beta=\gamma=0$  control independent approach

These can be straightforwardly generated using Generate_solutions.ipynb by inserting the corresponding parameters in there.

The large step-size h=0.8 might necessitate to use as initial condition another guess, e.g. from 0.4, which can be done by referring to its stored data



