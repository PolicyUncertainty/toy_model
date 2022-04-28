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
        model_dict["health_states"] = constr["health_states"]
    except KeyError:
        model_dict["health_states"] = np.arange(0, 2)

    try:
        model_dict["policy_states"] = constr["policy_states"]
    except KeyError:
        model_dict["policy_states"] = np.arange(0, 3)

    try:
        model_dict["num_quad_points"] = constr["N_QUAD"]
    except KeyError:
        model_dict["num_quad_points"] = 10
    try:
        model_dict["max_grid"] = constr["MAX_GRID"]
    except KeyError:
        model_dict["max_grid"] = 100
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
        param_dict["theta"] = np.append(
            np.random.uniform(1.5, 2.0), np.random.uniform(0, 0.5)
        )

    try:
        param_dict["lambda_"] = constr["TASTE_SHOCK"]
    except KeyError:
        param_dict["lambda_"] = np.random.uniform(2.2204e-16, 0.02)

    return param_dict
