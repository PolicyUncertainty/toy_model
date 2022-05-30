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


# @njit()
def conduct_backwards_induction(
    model_spec_params,
    utility_params,
    budget_params,
    discount_params,
    exog_gen_params,
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
    sigma = budget_params[2]
    beta = discount_params[0]

    # setup containers for the solution:

    # loop backwards over periods
    for t in range(Tbar - 1, -1, -1):
        # get possible states given period t
        subset_states = state_space[np.where(state_space[:, 0] == t)]
        # loop over all nodes in the state space given that we are in period t
        for state in subset_states:

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
                # compute choice specific next period wealth array. Does only depend on
                # current exogenous process state. So dependence here.
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

                choice_child_nodes = child_nodes[counter]

                choice_indexes_child = indices_child[counter]
                transition_probs = exog_gen_params[t, state[-1], :]

                # conduct egm
                ev, current_cons = conduct_egm(
                    next_period_wealth,
                    t,
                    value,
                    policy,
                    choice_child_nodes,
                    choice_indexes_child,
                    next_period_marg_budget,
                    quadw,
                    model_spec_params,
                    utility_params,
                    discount_params,
                    transition_probs,
                    indexer,
                    state_space,
                    indx_state,
                )

                # update policy container
                policy[indx_state, choice, 1, 1 : n_savings_grid + 1] = current_cons
                policy[indx_state, choice, 0, 1 : n_savings_grid + 1] = (
                    current_cons + savings_grid
                )
                policy[indx_state, choice, :, 0] = 0.0

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
                if con_envelope:
                    try:
                        # return value, policy
                        # print(indx_state, choice)
                        adjusted_value, adjusted_policy = secondary_envelope_wrapper(
                            value[indx_state, choice, :, : n_savings_grid + 1],
                            policy[indx_state, choice, :, : n_savings_grid + 1],
                            choice,
                            model_spec_params,
                            utility_params,
                        )
                        value[indx_state, choice, :, 1 : n_savings_grid + 1] = np.nan
                        policy[indx_state, choice, :, 1 : n_savings_grid + 1] = np.nan

                        value[
                            indx_state, choice, :, : adjusted_value.shape[1]
                        ] = adjusted_value
                        policy[
                            indx_state, choice, :, : adjusted_policy.shape[1]
                        ] = adjusted_policy
                    except:
                        breakpoint()
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
    num_choices = model_spec_params[4]
    return np.arange(num_choices)


@njit()
def get_child_nodes(state, state_space, poss_choices, indexer):
    """Based on a parent node in the state space as well as the
    node specific choice set, this function evaluates the nodes
    within the state space that can be reached from the current
    position in the decision tree.
    """
    indices = np.empty((poss_choices.shape[0], indexer.shape[-1]), dtype=np.int64)
    childs = np.empty(
        (poss_choices.shape[0], indexer.shape[-1], state_space.shape[1]), dtype=np.int64
    )
    child_state = np.empty_like(state)
    child_state[:] = state[:]
    # if the individual is already retired there is only one child node
    for counter, choice in enumerate(poss_choices):
        child_state[0] = state[0] + 1
        child_state[1] = choice
        for exog_process in range(indexer.shape[-1]):
            child_state[-1] = exog_process
            indices[counter, exog_process] = get_index_by_state(indexer, child_state)
            childs[counter, exog_process, :] = child_state

    return childs, indices


# @njit()
def conduct_egm(
    next_period_wealth,
    period,
    value,
    policy,
    choice_child_nodes,
    choice_indexes_child,
    next_period_marg_budget,
    quadw,
    model_spec_params,
    utility_params,
    discount_params,
    transition_probs,
    indexer,
    state_space,
    current_state_index,
):
    next_period_wealth_flatten = next_period_wealth.T.copy().reshape(
        next_period_wealth.shape[0] * next_period_wealth.shape[1]
    )
    num_exog_states = choice_child_nodes.shape[0]
    ev_exog_process_states = np.empty(
        (num_exog_states, next_period_wealth_flatten.shape[0])
    )
    marg_util_exog_states = np.empty(
        (num_exog_states, next_period_wealth_flatten.shape[0])
    )

    for exog_process in range(num_exog_states):
        child_state = choice_child_nodes[exog_process, :]
        next_period_choice_set = obtain_choice_set(child_state, model_spec_params)
        size_next_period_choice_set = next_period_choice_set.shape[0]
        indx_child_node = choice_indexes_child[exog_process]

        next_period_values_choices = np.empty(
            (size_next_period_choice_set, next_period_wealth_flatten.shape[0])
        )
        next_period_cons = np.empty(
            (size_next_period_choice_set, next_period_wealth_flatten.shape[0])
        )
        # loop over possible choices in the next period and compute
        #  the associated value functions!
        for counter, next_choice in enumerate(next_period_choice_set):
            if size_next_period_choice_set == 1:

                next_period_values_choices[0] = compute_next_period_value_(
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
                next_period_values_choices[counter, :] = compute_next_period_value_(
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

        # construct state specific continuation value
        ev_exog_process_states[exog_process, :] = expected_choice_value(
            next_period_values_choices, utility_params
        )
        # compute marginal utility of expected consumption in the next period
        marg_util_exog_states[exog_process, :] = compute_next_period_marg_util(
            next_period_cons, next_period_values_choices, utility_params
        )
    ev_flat = aggregate_exog_process(ev_exog_process_states, transition_probs)
    ev = aggregate_quadrature(ev_flat, quadw, model_spec_params)
    next_period_marginal_utility_flatten = aggregate_exog_process(
        marg_util_exog_states, transition_probs
    )

    next_period_marginal_utility = next_period_marginal_utility_flatten.reshape(
        # Size saving grid times num quadrature
        model_spec_params[3],
        model_spec_params[1],
    ).T.copy()
    # compute right hand side of the euler equation
    rhs = np.dot(quadw, next_period_marginal_utility * next_period_marg_budget)

    current_consumption = inv_marg_util(discount_params[0] * rhs, utility_params)
    if np.isnan(current_consumption).all() or current_state_index == 613:
        breakpoint()
    return ev, current_consumption


@njit()
def aggregate_exog_process(flat_wealth_array, transition_params):
    return np.dot(transition_params, flat_wealth_array)


@njit()
def expected_choice_value(next_period_value, utility_params):
    if next_period_value.shape[0] != 1:
        return logsum(next_period_value, utility_params)
    else:
        return next_period_value.flatten()


@njit()
def aggregate_quadrature(flat_array, quadw, model_spec_params):
    num_quad = quadw.shape[0]
    num_grid = model_spec_params[3]

    return np.dot(
        quadw.T,
        flat_array.reshape(num_grid, num_quad).T,
    )


@njit()
def compute_next_period_marg_util(next_period_cons, next_period_value, utility_params):
    choice_probs = calculate_choice_probs(next_period_value, utility_params)

    return (choice_probs * marg_util(next_period_cons, utility_params)).sum(axis=0)


@njit()
def calculate_choice_probs(next_period_value, utility_params):
    """Calculate the probability of choosing work in t+1
    for state worker given t+1 value functions"""
    lambda_ = utility_params[1]
    a = np.empty(next_period_value.shape[1])

    for j in range(next_period_value.shape[1]):
        a[j] = np.max(next_period_value[:, j])

    mxx = next_period_value - a
    return np.exp(mxx / lambda_) / np.sum(np.exp(mxx / lambda_), axis=0)


@njit()
def logsum(values, utility_params):
    """Calculate expected value function"""
    lambda_ = utility_params[1]

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
