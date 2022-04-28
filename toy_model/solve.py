import numpy as np

from toy_model.state_space import create_state_space


def solve(model_spec, param_spec, con_envelope=True):
    """This function wraps all steps that are necessary to solve the model."""
    # create savingsgrid
    savings_grid = np.linspace(0, model_spec.max_grid, model_spec.num_grid)

    # create state space
    state_space, indexer = create_state_space(model_spec)
