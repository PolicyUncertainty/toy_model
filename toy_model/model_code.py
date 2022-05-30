import numpy as np
from numba import njit


@njit()
def util(consumption, utility_params, choice):
    """This function implements a simple cobb douglas utility function."""
    out = consumption ** utility_params[0] + (
        (24 * 5) - get_discrete_choice_values(choice)
    ) ** (1 - utility_params[0])
    return out


@njit()
def marg_util(cons, utility_params):
    """marginal utility"""
    return utility_params[0] * (cons ** (utility_params[0] - 1))


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
    num_grid = savings_grid.shape[0]
    num_quad = shocks.shape[0]
    period, _, health_state, policy_state = state
    age = 20 + period
    hourly_wage = np.exp(hourly_wage_systematic(age, budget_params) + shocks)
    health_costs = calc_health_care_costs(
        health_state, policy_state, model_spec_params, budget_params
    )

    total_income = get_weekly_income(hourly_wage, choice)
    compound_assets = (1 + budget_params[4]) * savings_grid
    out = (
        (np.ones((num_grid, num_quad)) * total_income).T
        + compound_assets
        - health_costs
    )
    out_filtered = np.where(out >= 0, out, 100.0)
    return out_filtered


@njit()
def get_weekly_income(hourly_wage, choice):
    if choice != 0:
        return get_discrete_choice_values(choice) * hourly_wage
    else:
        return np.array([100.0])


@njit()
def get_discrete_choice_values(choice):
    return np.array([0, 20, 38])[choice]


@njit()
def calc_health_care_costs(
    health_state, policy_state, model_spec_params, budget_params
):
    health_care_costs = model_spec_params[7]
    covered_share = budget_params[3:5][policy_state]
    return health_state * health_care_costs * (1 - covered_share)


@njit()
def marg_budget(budget_params, num_grid, num_quad):
    """Derivative by saving."""
    r = budget_params[0]
    return np.ones((num_quad, num_grid)) * (1 + r)
