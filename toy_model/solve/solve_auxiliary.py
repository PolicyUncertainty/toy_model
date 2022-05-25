"""This model contains auxiliary files that are utilized to solve
the model.
"""
import numpy as np
from numba import njit

MISSING_INT = -9999  # noqa


@njit()
def create_state_space(model_spec_params):
    """This function creates two objects: the state space and the indexer.
    For the current status of the model this seems to be overkill. Never-
    theless it could be userful to implement this kind of technique before
    the model gets extended since it means that we do not have to adjust to
    many things later on.
    """

    num_choices = model_spec_params[6]
    num_periods = model_spec_params[0]
    max_ret_age = model_spec_params[1]
    min_ret_age = model_spec_params[2]
    # set up the indexer object ( not necessary now since the model is
    # small but maybe useful in the future)
    shape = (
        num_periods,
        num_choices,
        max_ret_age + 1,
    )
    indexer = np.full(shape, -9999, dtype=np.int64)
    data = []
    i = 0
    # for each of the periods
    for period in range(num_periods):
        # for each of the choices an individual could have taken in period t-1
        for last_period_decision in range(num_choices):

            if period <= min_ret_age:
                # individuals can not retire before the prespecified age

                if last_period_decision > 1:
                    continue
                else:
                    indexer[period, last_period_decision, 0] = i

                    row = [period, last_period_decision, 0]
                    i += 1
                    data.append(row)
            else:

                # it is no possible to be working after the max_retirement age
                # (at least in the model)
                if (period > max_ret_age) & (last_period_decision < 2.0):
                    continue
                # in the "retirement" phase the agent can have decided to work or being
                # unemployed
                elif (period <= max_ret_age) & (last_period_decision < 2.0):
                    indexer[period, last_period_decision, 0] = i

                    row = [period, last_period_decision, 0]

                    i += 1
                    data.append(row)
                # the individual chose retirement in the previous period, we
                # have to add information about the actual time period in which
                #  the individual retired since affects the amount of pension
                elif last_period_decision == 2.0:

                    for ret_period in range(min_ret_age, min(period, max_ret_age + 1)):

                        indexer[period, last_period_decision, ret_period] = i

                        row = [period, last_period_decision, ret_period]

                        i += 1
                        data.append(row)

    states = np.array(data, dtype=np.int64)
    return states, indexer


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
    max_ret_age = model_spec_params[1]
    min_ret_age = model_spec_params[2]
    num_choices = model_spec_params[6]
    Tbar = model_spec_params[0] - 1
    n_savings_grid = model_spec_params[5]

    # create arrays that are "too long", necessary for the secondary envelope
    policy = np.empty(
        (state_space.shape[0], num_choices, 2, int(1.1 * n_savings_grid + 1))
    )
    policy[:] = np.nan

    value = np.empty(
        (state_space.shape[0], num_choices, 2, int(1.1 * n_savings_grid + 1))
    )
    value[:] = np.nan

    # set values for the terminal period (agent can not be working or unemployed
    # and there is no continuation value)

    # determine indices of the terminal periods:
    indxs = indexer[Tbar, 2, min_ret_age : max_ret_age + 1]
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
