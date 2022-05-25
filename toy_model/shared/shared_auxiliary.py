"""This module contains auxiliary variables that are used by the simulation and
the solve processes.
"""
import collections

import numpy as np
from numba import njit


def dict_to_namedtuple_spec(dictionary, name):
    """Coverts non-nested dictionary to namedtuple"""

    return collections.namedtuple(f"{name}_spec", dictionary.keys())(**dictionary)


def process_model_spec(model_spec):
    pass


def set_up_solution_container():
    pass


@njit()
def custom_interp1d(z, x_new, y_new):
    """This function conducts a 1 dimensional spline interpolation."""
    y = []
    for i in z:
        idx = search(0, len(x_new), x_new, i)
        nom1 = x_new[idx + 1] - i
        nom2 = i - x_new[idx]
        denom = (x_new[idx + 1] - x_new[idx]) + 2.220446049250313e-16
        y.append(y_new[idx] * (nom1 / denom) + y_new[idx + 1] * (nom2 / denom))

    return np.array(y)


@njit()
def search(min_indx, grid_length, sorted_grid, element):
    """This function searches the element on the specific grid
    that is used for the interpolation.
    """

    # if the element is outside the bounds
    # use the border elements for interpolation

    if element <= sorted_grid[0]:
        return 0
    elif element >= sorted_grid[grid_length - 2]:
        return grid_length - 2

    # if within bounds apply binary search algorithm

    # Divide grid length by two and round down
    aux = grid_length // 2

    # while aux != 0 continue searching
    while aux:
        # create a new candidate
        candidate = min_indx + aux

        # if candidate element is smaller or equal to element update min_indx
        if sorted_grid[candidate] <= element:
            min_indx = candidate

        # update grid_length (since we can reject half of the grid candidates)
        grid_length -= aux
        aux = grid_length // 2

    return min_indx


@njit()
def filter_array(arr):
    """This function grabs only non nan entries from the policy/value array."""
    for k, i in enumerate(arr[1]):

        if np.isnan(i):
            return arr[:, :k]
        else:
            continue
    return arr
