import numpy as np


def create_state_space(model_spec):
    shape = (
        model_spec.num_periods,
        model_spec.choices.shape[0],
        model_spec.health_states.shape[0],
        model_spec.policy_states.shape[0],
    )
    indexer = np.full(shape, -9999, dtype=np.int64)
    data = []
    i = 0

    # for each of the periods
    for period in range(model_spec.num_periods):
        # for each of the choices an individual could have taken in period t-1
        for last_period_decision in model_spec.choices:
            # for all possible health states
            for health_state in model_spec.health_states:
                # for possible policy states
                for policy_state in model_spec.policy_states:
                    indexer[
                        period, last_period_decision, health_state, policy_state
                    ] = i

                    row = [period, last_period_decision, health_state, policy_state]
                    i += 1
                    data.append(row)

    states = np.array(data, dtype=np.int64)
    return states, indexer
