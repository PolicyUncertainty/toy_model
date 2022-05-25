"""Simplified solving algorithm for our toy model."""
import numpy as np
from numba import njit

from toy_model.model_code import budget
from toy_model.model_code import inv_marg_util
from toy_model.model_code import marg_budget
from toy_model.model_code import marg_util
from toy_model.model_code import util
from toy_model.shared.shared_auxiliary import custom_interp1d
from toy_model.shared.shared_auxiliary import filter_array
from toy_model.solve.secondary_envelope import secondary_envelope_wrapper
from toy_model.state_space import get_index_by_state

# from numba.core.errors import TypingError


@njit()
def conduct_backwards_induction(
    model_spec_params,
    utility_params,
    budget_params,
    discount_params,
    state_space,
    indexer,
    policy,
    value,
    savings_grid,
    quadstnorm,
    quadw,
    con_envelope,
):
    """This function conducts the backwards induction."""
    Tbar = model_spec_params[0] - 1
    num_quad = quadw.shape[0]
    n_savings_grid = savings_grid.shape[0]
    sigma = budget_params[4]
    beta = discount_params[0]

    # setup containers for the solution:

    # loop backwards over periods
    for t in range(Tbar - 1, -1, -1):
        # get possible states given period t
        subset_states = state_space[np.where(state_space[:, 0] == t)]
        # loop over all nodes in the state space given that we are in period t
        for state in subset_states:

            period, last_period_choice, status = state
            # determine indx of the state
            indx_state = get_index_by_state(indexer, state)  # determine choice set
            choice_set = obtain_choice_set(state, model_spec_params)
            # determine child nodes within the state space that the individual
            # can reach based on the available choices
            child_nodes, indices_child = get_child_nodes(
                state, state_space, choice_set, indexer
            )
            # size_choice_set = choice_set.shape[0]

            for counter, choice in enumerate(choice_set):

                child_node = child_nodes[counter]

                indx_child = indices_child[counter]
                next_period_choice_set = obtain_choice_set(
                    child_node, model_spec_params
                )
                # compute choice specific next period wealth array
                next_period_wealth = budget(
                    state,
                    savings_grid,
                    choice,
                    quadstnorm * sigma,
                    model_spec_params,
                    utility_params,
                    budget_params,
                )

                # marginal budget can also be determined outside of the egm step
                # (since end of period assets as well as the return do not depend
                # on the previous decisions)
                next_period_marg_budget = marg_budget(
                    budget_params, n_savings_grid, num_quad
                )

                # conduct egm
                next_period_value, current_cons = conduct_egm(
                    next_period_wealth,
                    t,
                    value,
                    policy,
                    indx_child,
                    next_period_choice_set,
                    next_period_marg_budget,
                    quadw,
                    model_spec_params,
                    utility_params,
                    discount_params,
                )

                # update policy container
                policy[indx_state, choice, 1, 1 : n_savings_grid + 1] = current_cons
                policy[indx_state, choice, 0, 1 : n_savings_grid + 1] = (
                    current_cons + savings_grid
                )
                policy[indx_state, choice, :, 0] = 0.0
                # construct state specific continuation value
                ev = expected_value_(
                    next_period_value, quadw, utility_params, model_spec_params
                )

                # update value container
                value[indx_state, choice, 1, 1 : n_savings_grid + 1] = (
                    util(current_cons, utility_params, choice) + beta * ev
                )
                value[indx_state, choice, 0, 1 : n_savings_grid + 1] = (
                    current_cons + savings_grid
                )
                value[indx_state, choice, 1, 0] = ev[0]
                value[indx_state, choice, 0, 0] = 0.0
                # conduct secondary envelope
                if con_envelope and next_period_choice_set.shape[0] != 1:

                    # return value, policy
                    # print(indx_state, choice)
                    adjusted_value, adjusted_policy = secondary_envelope_wrapper(
                        value[indx_state, choice, :, : n_savings_grid + 1],
                        policy[indx_state, choice, :, : n_savings_grid + 1],
                        choice,
                        utility_params,
                        model_spec_params,
                    )
                    value[indx_state, choice, :, 1 : n_savings_grid + 1] = np.nan
                    policy[indx_state, choice, :, 1 : n_savings_grid + 1] = np.nan

                    value[
                        indx_state, choice, :, : adjusted_value.shape[1]
                    ] = adjusted_value
                    policy[
                        indx_state, choice, :, : adjusted_policy.shape[1]
                    ] = adjusted_policy
                    # except (IndexError, TypingError, AttributeError, ValueError):
                    #    tb = traceback.format_exc()
                    #    print(value[indx_state, choice, :, : n_savings_grid + 1])
                    #    print(policy[indx_state, choice, :, : n_savings_grid + 1])
                    #    print(indx_state, choice)
                    #    print(tb)
                    # return value, policy, indexer
    return value, policy


@njit()
def obtain_choice_set(state, model_spec_params):
    num_choices = model_spec_params[3]
    return np.arange(num_choices)


@njit()
def get_child_nodes(state, state_space, poss_choices, indexer):
    """Based on a parent node in the state space as well as the
    node specific choice set, this function evaluates the nodes
    within the state space that can be reached from the current
    position in the decision tree.
    """
    indices = np.empty(0, dtype=np.int64)
    childs = np.empty((poss_choices.shape[0], state_space.shape[1]), dtype=np.int64)
    period, last_period_choice, status = state
    # if the individual is already retired there is only one child node
    if status != 0:
        indx = indexer[period + 1, last_period_choice, status]
        indices = np.append(indices, np.array([indx], dtype=np.int64))
        childs[0] = state_space[indx]

        # if not, there are 2 to three different nodes
    else:
        for counter, i in enumerate(poss_choices):
            # if the individual hasn't entered retirement yet, status=0
            if i in np.array([0, 1], dtype=np.int64):
                indx = indexer[period + 1, i, 0]
                childs[counter] = state_space[indx]
            else:
                indx = indexer[period + 1, i, period]
                childs[counter] = state_space[indx]

            indices = np.append(indices, np.array([indx], dtype=np.int64))

    return childs, indices


@njit()
def conduct_egm(
    next_period_wealth,
    period,
    value,
    policy,
    indx_child_node,
    next_period_choice_set,
    next_period_marg_budget,
    quadw,
    model_spec_params,
    utility_params,
    discount_params,
):
    next_period_wealth_flatten = next_period_wealth.T.copy().reshape(
        next_period_wealth.shape[0] * next_period_wealth.shape[1]
    )
    # set up container for the values and the associated consumption!
    size_next_period_choice_set = next_period_choice_set.shape[0]
    next_period_values = np.empty(
        (size_next_period_choice_set, next_period_wealth_flatten.shape[0])
    )
    next_period_cons = np.empty(
        (size_next_period_choice_set, next_period_wealth_flatten.shape[0])
    )
    # loop over possible choices in the next period and compute
    #  the associated value functions!
    for counter, next_choice in enumerate(next_period_choice_set):
        if size_next_period_choice_set == 1:

            next_period_values[0] = compute_next_period_value_(
                next_period_wealth_flatten,
                next_choice,
                period,
                value,
                indx_child_node,
                model_spec_params,
                utility_params,
                discount_params,
            )
            next_period_cons[0] = compute_next_period_cons(
                policy, next_choice, indx_child_node, next_period_wealth_flatten
            )
        else:
            next_period_values[counter, :] = compute_next_period_value_(
                next_period_wealth_flatten,
                next_choice,
                period,
                value,
                indx_child_node,
                model_spec_params,
                utility_params,
                discount_params,
            )
            next_period_cons[counter, :] = compute_next_period_cons(
                policy, next_choice, indx_child_node, next_period_wealth_flatten
            )

    # compute marginal utility of expected consumption in the next period
    next_period_marginal_utility_flatten = compute_next_period_marg_util(
        next_period_cons, next_period_values, utility_params
    )
    next_period_marginal_utility = next_period_marginal_utility_flatten.reshape(
        # Size saving grid times num quadrature
        model_spec_params[3],
        model_spec_params[1],
    ).T.copy()
    # compute right hand side of the euler equation
    rhs = np.dot(quadw, next_period_marginal_utility * next_period_marg_budget)

    current_consumption = inv_marg_util(discount_params[0] * rhs, utility_params)

    return next_period_values, current_consumption


@njit()
def expected_value_(next_period_value, quadw, utility_params, model_spec_params):
    num_quad = quadw.shape[0]
    num_grid = model_spec_params[3]

    if next_period_value.shape[0] != 1:

        return np.dot(
            quadw.T,
            logsum(next_period_value, utility_params).reshape(num_grid, num_quad).T,
        )
    else:
        return np.dot(quadw.T, next_period_value.reshape(num_grid, num_quad).T)


@njit()
def compute_next_period_marg_util(next_period_cons, next_period_value, utility_params):
    choice_probs = calculate_choice_probs(next_period_value, utility_params)

    return (choice_probs * marg_util(next_period_cons, utility_params)).sum(axis=0)


@njit()
def calculate_choice_probs(next_period_value, utility_params):
    """Calculate the probability of choosing work in t+1
    for state worker given t+1 value functions"""
    lambda_ = utility_params[2]
    a = np.empty(next_period_value.shape[1])

    for j in range(next_period_value.shape[1]):
        a[j] = np.max(next_period_value[:, j])

    mxx = next_period_value - a
    return np.exp(mxx / lambda_) / np.sum(np.exp(mxx / lambda_), axis=0)


@njit()
def logsum(values, utility_params):
    """Calculate expected value function"""
    lambda_ = utility_params[2]

    a = np.empty(values.shape[1])

    for j in range(values.shape[1]):
        a[j] = np.max(values[:, j])

    mxx = values - a
    return a + lambda_ * np.log(np.sum(np.exp(mxx / lambda_), axis=0))


@njit()
def compute_next_period_value_(
    next_period_wealth,
    next_choice,
    period,
    value,
    indx_child_node,
    model_spec_params,
    utility_params,
    discount_params,
):
    # In the last period, the agent consumes everything that she has
    # therefore, the value of each state in the last period is equal
    # the float utility
    if period + 1 == model_spec_params[0] - 1:
        return util(next_period_wealth, utility_params, next_choice)
    else:
        return value_function(
            indx_child_node,
            next_choice,
            next_period_wealth,
            value,
            utility_params,
            discount_params,
        )


@njit()
def value_function(
    index_child, next_choice, next_period_wealth, value, utility_params, discount_params
):
    beta = discount_params[0]

    res = np.empty(next_period_wealth.shape)
    # Mark constrained region
    # credit constraint between 1st (M_{t+1) = 0) and second point (A_{t+1} = 0)
    next_period_value_arr = filter_array(value[index_child, next_choice])
    mask = next_period_wealth < next_period_value_arr[1, 0]
    # Calculate t+1 value function in the constrained region
    res[mask] = (
        util(next_period_wealth[mask], utility_params, next_choice)
        + beta * next_period_value_arr[0, 1]
    )
    # Calculate t+1 value function in non-constrained region
    # inter- and extrapolate
    res[~mask] = custom_interp1d(
        next_period_wealth[~mask],
        next_period_value_arr[0, :],
        next_period_value_arr[1, :],
    )

    return res


@njit()
def compute_next_period_cons(policy, next_choice, indx_child_node, next_period_wealth):
    """This function computes the optimal consumption in the following period
    given the pre-specified level of wealth in the next period. Tthe"""
    # create an array for the

    # clean nan values
    next_period_policy_arr = filter_array(policy[indx_child_node, next_choice])
    # interpolate consumption
    cons_t1 = custom_interp1d(
        next_period_wealth,
        next_period_policy_arr[0, :],
        next_period_policy_arr[1, :],
    )

    #
    return cons_t1
