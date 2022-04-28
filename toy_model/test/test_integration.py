from toy_model.solve import solve
from toy_model.test.random_init import generate_random_model_spec


def test1():
    """This test creates 10 random model specifications and runs
    solves the associated model.
    """
    for _ in range(10):
        model_spec, param_spec = generate_random_model_spec()
        _, _ = solve(model_spec, param_spec)
