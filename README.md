# skeleton_priors

All the loss functions I have written are in losses.py. Classic optimization on synthetic data is done with optimization_synthetic.py and on real data with optimization_real.py.
The function in optimization_parameters.py optimizes for the model parameters instead of 3D positions of the points. The function in optimization_real.py has the option to initialize with the forward kinematics model.
