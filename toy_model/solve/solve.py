"""This module contains the function that solves the economic model."""
import numpy as np
import scipy.stats as scps
from scipy.special.orthogonal import ps_roots

from toy_model.solve.backwards_induction import conduct_backwards_induction
from toy_model.state_space import create_container
from toy_model.state_space import create_state_space


def solve(model_spec, param_spec, exog_gen_params, con_envelope=True):
    """This function wraps all steps that are necessary to solve the model."""
    # create savingsgrid
    model_spec_params = np.array(
        [
            model_spec.num_periods,  # 0
            model_spec.num_quad_points,  # 1
            model_spec.max_grid,  # 2
            model_spec.num_grid,  # 3
            model_spec.choices.shape[0],  # 4
            model_spec.num_health_states,  # 5
            model_spec.num_policy_states,  # 6
            model_spec.total_hours,  # 7
            model_spec.health_costs,
        ],
        dtype=np.int32,
    )

    savings_grid = np.linspace(0, model_spec.max_grid, model_spec.num_grid)
    # create state space
    state_space, indexer = create_state_space(model_spec)
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
        np.append(
            np.array(
                [
                    param_spec.interest_rate,  # 0
                    param_spec.unemp,  # 1
                    param_spec.sigma,  # 2
                ],
                dtype=np.float64,
            ),
            param_spec.health_cost_policy.flatten(),
        ),  # 3, 4
        param_spec.wage_coeff.flatten(),  # 5, 6, 7
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
