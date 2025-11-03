# Code for the paper Variational integrators for a new Lagrangian approach to optimal control affine systems with a quadratic Lagrange term

This code can be used to generate the low-thrust orbitral transfer optimal control problem treated as an example in the paper 'Variational integrators for a new Lagrangian approach to optimal control affine systems with a quadratic Lagrange term' by Konopik et al, using variational integrators derved from a new Lagrangian approach to solving optimal control problems.



It consists of two parts:
1. A program that generates a dataset of the optimal control problem 
-- for a given set of parameters
-- for the standard approach, the new approaches

2. The program that creates the data for the performance study
3. The script that generates the figures of the paper once all data sets are generated


How to use the files.

1. Run 'Direct_approaches_solver.ipynb':
-- generate data by running the full jupyter file
-- To generate all data, the elements  'good_initial_guess', 'alpha_choice','gamma_choice','iterator' need to be used with the follong values in all combinations
-- 'good_initial_guess': 'True', 'False'
-- '(alpha_choice,gamma_choice)' = (1,1), (1,0.5), (0.5,1)
-- iterator= 0,1,...,11

2.  Run 'performane_data_generation.ipynb'
-- generates the performance data for the function evaluation

3. Run 'indirect_solution.ipynb' (optional)
-- generates the reference indirect solution. reference solution is already supplied as 'reference_polar.pkl'

3. Run 'plotting_data_low_thrust_study.ipynb'
-- generates the figures

