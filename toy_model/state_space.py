import numpy as np
from numba import njit


def create_state_space(model_spec):
    shape = (
        model_spec.num_periods,
        model_spec.choices.shape[0],
        model_spec.num_policy_states,
        model_spec.num_health_states,
    )
    indexer = np.full(shape, -9999, dtype=np.int64)
    data = []
    i = 0

    # for each of the periods
    for period in range(model_spec.num_periods):
        # for each of the choices an individual could have taken in period t-1
        for last_period_decision in model_spec.choices:
            # for possible policy states
            for policy_state in range(model_spec.num_policy_states):
                # for all possible health states
                for health_state in range(model_spec.num_health_states):
                    indexer[
                        period, last_period_decision, policy_state, health_state
                    ] = i

                    row = [period, last_period_decision, policy_state, health_state]
                    i += 1
                    data.append(row)

    states = np.array(data, dtype=np.int64)
    return states, indexer


def get_index_by_state(indexer, state):
    return indexer[state[0], state[1], state[2], state[3]]


@njit()
def create_container(
    state_space,
    indexer,
    savings_grid,
    model_spec_params,
):
    """This function creates the container in that the model solution is
    stored.
    """
    num_choices = indexer.shape[1]
    n_savings_grid = savings_grid.shape[0]

    # create arrays that are "too long", necessary for the secondary envelope
    policy = np.empty(
        (state_space.shape[0], num_choices, 2, int(1.1 * n_savings_grid + 1))
    )
    policy[:] = np.nan

    value = np.empty(
        (state_space.shape[0], num_choices, 2, int(1.1 * n_savings_grid + 1))
    )
    value[:] = np.nan
    policy, value = assign_bequest(
        indexer, policy, value, savings_grid, model_spec_params
    )
    return policy, value


@njit()
def assign_bequest(indexer, policy, value, savings_grid, model_spec_params):
    Tbar = model_spec_params[0] - 1
    n_savings_grid = savings_grid.shape[0]

    # set values for the terminal period (agent can not be working or unemployed
    # and there is no continuation value)

    # determine indices of the terminal periods:
    indxs = indexer[Tbar, :, :, :].flatten()
    for inx in indxs:
        policy[inx, 2, 0, : n_savings_grid + 1] = np.append([0], savings_grid)
        policy[inx, 2, 1, : n_savings_grid + 1] = np.append([0], savings_grid)
        value[inx, 2, 0, : n_savings_grid + 1] = np.append(
            [0], np.zeros(n_savings_grid)
        )
        value[inx, 2, 1, : n_savings_grid + 1] = np.append(
            [0], np.zeros(n_savings_grid)
        )
    return policy, value
