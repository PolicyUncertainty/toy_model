"""The module includes an integration and a regression test"""
import numpy as np

from toy_model.solve.solve import solve
from toy_model.test.random_init import generate_random_model_spec


def test1():
    """This test creates 10 random model specifications and runs
    solves the associated model.
    """
    for _ in range(10):
        model_spec, param_spec = generate_random_model_spec()
        healt_states_probs = np.zeros(
            (
                model_spec.num_periods,
                model_spec.num_health_states,
                model_spec.num_health_states,
            )
        )
        for t in range(model_spec.num_periods):
            for state in range(2):
                healt_states_probs[t, state, 0] = np.random.uniform()
                healt_states_probs[t, state, 1] = 1 - healt_states_probs[t, state, 0]
        _, _ = solve(model_spec, param_spec, healt_states_probs)
