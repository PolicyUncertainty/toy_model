"""This file contains functions that are used to create random model specifi-
cations for test purposes.
"""
import numpy as np

from toy_model.pre_processing import dict_to_namedtuple_spec


def generate_random_model_spec(constr_model=None, constr_param=None):
    """This function generates a random model specification."""
    if constr_model is None:
        constr_model = dict()
    if constr_param is None:
        constr_param = dict()
    # first model specs
    model_dict = process_model_constr_inputs(constr_model)

    # second param specs
    param_dict = process_param_constr_inputs(constr_param)

    return dict_to_namedtuple_spec(model_dict, "model"), dict_to_namedtuple_spec(
        param_dict, "param"
    )


def process_model_constr_inputs(constr):
    model_dict = dict()
    try:
        model_dict["num_periods"] = constr["PERIODS"]
    except KeyError:
        model_dict["num_periods"] = np.random.randint(30, 60)

    try:
        model_dict["num_health_states"] = constr["num_health_states"]
    except KeyError:
        model_dict["num_health_states"] = 2

    try:
        model_dict["num_policy_states"] = constr["num_policy_states"]
    except KeyError:
        model_dict["num_policy_states"] = 2

    try:
        model_dict["health_costs"] = constr["health_costs"]
    except KeyError:
        model_dict["health_costs"] = 100

    # try:
    #     model_dict["total_hours"] = constr["total_hours"]
    # except KeyError:
    #     model_dict["total_hours"] = 16 * 5

    try:
        model_dict["num_quad_points"] = constr["N_QUAD"]
    except KeyError:
        model_dict["num_quad_points"] = 10
    try:
        model_dict["max_grid"] = constr["MAX_GRID"]
    except KeyError:
        model_dict["max_grid"] = 10_000
    try:
        model_dict["num_grid"] = constr["N_GRID"]
    except KeyError:
        model_dict["num_grid"] = np.random.randint(100, 500)

    # so far the codebase isn't flexible enough to allow to specify more
    # choices, therefore this will be fixed from now on
    model_dict["choices"] = np.array([0, 1, 2])

    return model_dict


def process_param_constr_inputs(constr):
    param_dict = dict()
    try:
        param_dict["beta"] = constr["DISCOUNT"]
    except KeyError:
        param_dict["beta"] = np.random.uniform(0.7, 0.999)

    try:
        param_dict["theta"] = constr["UTILITY_PARAMS"]
    except KeyError:
        param_dict["theta"] = np.random.uniform(0, 1)

    try:
        param_dict["wage_coeff"] = constr["WAGE_PARAMS"]
    except KeyError:
        param_dict["wage_coeff"] = np.append(
            np.append(np.random.uniform(1.5, 2.0), np.random.uniform(0.01, 0.03)),
            -np.random.uniform(0, 0.00004),
        )

    try:
        param_dict["health_cost_policy"] = constr["POLICY_PARAMS"]
    except KeyError:
        param_dict["health_cost_policy"] = np.append(
            np.random.uniform(0, 0.5), np.random.uniform(0, 0.5)
        )

    try:
        param_dict["unemp"] = constr["UNEMP_PARAMS"]
    except KeyError:
        param_dict["unemp"] = 200

    try:
        param_dict["interest_rate"] = constr["INTEREST_RATE"]
    except KeyError:
        param_dict["interest_rate"] = 0.05

    try:
        param_dict["sigma"] = constr["WAGE_SHOCK"]
    except KeyError:
        param_dict["sigma"] = np.random.uniform(2.2204e-16, 0.02)

    try:
        param_dict["lambda_"] = constr["TASTE_SHOCK"]
    except KeyError:
        param_dict["lambda_"] = np.random.uniform(2.2204e-16, 0.02)

    return param_dict
