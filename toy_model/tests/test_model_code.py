from toy_model.model_code import utility


def test_utility():
    cons = 9
    gamma = 1/2
    budget = 16
    working = 7
    util = utility(cons, working, budget, gamma)
    assert util == 6
