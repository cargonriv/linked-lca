"""Module containing configuration parameters for experiments."""
metaparams = 7  # number of meta-parameters that are empirical to the dff traces.
PRINT_FREQ = 1  # sparsity calculations' print frequency w/ modulus operandi.
KERNEL_SIZE = 9  # Kernel value of the dimensions for convolutional features.
WINDOW_SIZE = 7  # number of frames for each windowed video batch.
batch_split = 6  # number of splits for each window video batch.
LCA_ITERS = 200  # number of LCA timesteps per forward pass.
FEATURES = 50  # number of dictionary features to learn.
exp_star = 15  # start index of exp cell slicing.
exp_end = 21  # ending index of cell slicing.
LAMBDA = 0.5  # neuron activation threshold.
STRIDE = 2  # LCA convolutional stride.
EPOCHS = 1  # periods of training.
TAU = 200  # LCA time constant.
i = 0
