import numpy as np
from numba import njit

HOURS = [0, 20, 38]


@njit()
def util(consumption, utility_params, choice):
    """This function implements a simple cobb douglas utility function."""
    out = consumption ** utility_params[0] + (24 - HOURS[choice]) ** (
        1 - utility_params[0]
    )
    return out


@njit()
def marg_util(cons, utility_params):
    """marginal utility"""
    return utility_params[0](cons ** (utility_params[0] - 1))


@njit()
def inv_marg_util(m_u, utility_params):
    """Inverse marginal utility"""
    return (m_u / utility_params[0]) ** (1 / utility_params[0] - 1)


@njit()
def hourly_wage_systematic(age, budget_params):
    """Calculate the hourly wage. So far only dependent on age."""
    wage_params = budget_params[5:8]
    out = wage_params[0] + wage_params[1] * age + wage_params[2] * age**2
    return out


@njit()
def budget(
    state,
    savings_grid,
    choice,
    shocks,
    model_spec_params,
    utility_params,
    budget_params,
):
    """Get new assets"""
    period, _, health_state, policy_state = state
    age = 20 + period
    hourly_wage = hourly_wage_systematic(age, budget_params) + shocks
    health_costs = calc_health_care_costs(
        health_state, policy_state, model_spec_params, budget_params
    )
    total_income = HOURS[choice] * hourly_wage
    compound_assets = (1 + budget_params[4]) * savings_grid
    out = compound_assets + total_income - health_costs
    return out


@njit()
def calc_health_care_costs(
    health_state, policy_state, model_spec_params, budget_params
):
    health_care_costs = model_spec_params[4]
    covered_share = budget_params[5:7][policy_state]
    return health_state * health_care_costs * (1 - covered_share)


@njit()
def marg_budget(budget_params, num_grid, num_quad):
    """Derivative by saving."""
    r = budget_params[2]
    return np.ones((num_quad, num_grid)) * (1 + r)
