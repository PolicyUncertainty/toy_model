import pytest

from toy_model.model_code import util


@pytest.fixture(scope="module")
def input_data():
    cons = 9
    gamma = 1 / 2
    budget = 16
    working = 7
    return cons, gamma, budget, working


def test_utility(input_data):
    cons, gamma, budget, working = input_data
    util = util(cons, working, budget, gamma)
    assert util == 6
