"""This module contains the function that solves the economic model."""
import numpy as np
import scipy.stats as scps
from retpy.solve.backwards_induction import conduct_backwards_induction
from retpy.solve.solve_auxiliary import create_container
from retpy.solve.solve_auxiliary import create_state_space
from scipy.special.orthogonal import ps_roots


def solve(model_spec, param_spec, con_envelope=True):
    """This function wraps all steps that are necessary to solve the model."""
    # create savingsgrid
    model_spec_params = np.array(
        [
            model_spec.num_periods,  # 0
            model_spec.max_ret_age,  # 1
            model_spec.min_ret_age,  # 2
            model_spec.num_quad_points,  # 3
            model_spec.max_grid,  # 4
            model_spec.num_grid,  # 5
            model_spec.choices.shape[0],  # 6
        ],
        dtype=np.int32,
    )

    savings_grid = np.linspace(0, model_spec_params[4], model_spec_params[5])
    # create state space
    state_space, indexer = create_state_space(model_spec_params)
    # prepare quadrature
    quadstnorm = scps.norm.ppf(ps_roots(model_spec.num_quad_points)[0])
    quadw = ps_roots(model_spec.num_quad_points)[1]

    # create solution container
    pol, val = create_container(state_space, indexer, savings_grid, model_spec_params)
    utility_params = np.array(
        [
            param_spec.theta[0],  # 0
            param_spec.theta[1],  # 1
            param_spec.lambda_,  # 2
        ],
        dtype=np.float64,
    )
    budget_params = np.append(
        np.array(
            [
                param_spec.share,  # 0
                param_spec.reduc,  # 1
                param_spec.rf,  # 2
                param_spec.unemp,  # 3
                param_spec.sigma,  # 4
            ],
            dtype=np.float64,
        ),
        param_spec.coef.flatten(),
    )

    discount_params = np.array([param_spec.beta], dtype=np.float64)

    # conduct backwards induction and upper envelope
    value, policy = conduct_backwards_induction(
        model_spec_params,
        utility_params,
        budget_params,
        discount_params,
        state_space,
        indexer,
        pol,
        val,
        savings_grid,
        quadstnorm,
        quadw,
        con_envelope,
    )
    return value, policy
