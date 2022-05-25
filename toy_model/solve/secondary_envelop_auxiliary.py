"""This file contains the auxiliary functions that are used to
conduct the secondary envelope.
"""
import numpy as np
from numba import njit
from numba import prange
from numba.typed import List
from retpy.shared.shared_auxiliary import custom_interp1d


@njit()
def chop(a, repeat):
    """This function does two crucial things:
    1. it loops over all values within the grid and determines discontinuities
    where mj > mj+1.

    2. According to the discontinuties the functions splits the grid in sections.

    If repeat==False, the discontinuity point is only part of one of the splitted
    section arrays.

    Version without empty arrays (at least as long as repeat=False)
    """

    chopped_array = List()
    cut_points = List()
    k = 0
    for counter, i in enumerate(a[0]):
        # check if there is an discontinuity
        if counter + 1 < a.shape[1]:

            if i > a[0, counter + 1]:
                # chop array+
                cut_points.append(counter)
                if k != counter:

                    chopped_array.append(a[:, k : counter + 1])
                if repeat:
                    k = np.min(np.where(a[0, counter:] < i)[0]) + counter
                    chopped_array.append(a[:, counter : k + 1])
                else:
                    k = np.min(np.where(a[0, counter:] < i)[0]) + counter
        else:
            # this if condition is important since otherwise arrays
            # of size 1 are added
            if a[:, k:].shape[1] > 1:
                chopped_array.append(a[:, k:])
            else:
                continue
    return cut_points, chopped_array


@njit()
def check_chopped_parts(chopped_array):

    """This function returns the x any y axis lengths of the chopped grid sections."""
    non_empty = List()
    for i in chopped_array:
        if i.shape[0] > 0:
            non_empty.append(i)
        else:
            continue
    return non_empty


@njit()
def interpolation_upper_envelop(z, non_empty):
    """This function uses the customized spline interpolation function to
    inter/extrapolate y values for all z values based on the sections of
    the grid in non_empty.
    """

    # set up an result array
    interp_rslts = np.empty((len(non_empty), z.shape[0]))

    extrapolation_filter = np.empty((len(non_empty), z.shape[0]))
    # loop over all sections
    for counter, array in enumerate(non_empty):
        # set each row in the rslt array equal to the interpolated values
        interp_rslts[counter] = custom_interp1d(z, array[0], array[1])
        # determine which of the values are actually not within the specific section
        extrapolation_filter[counter] = np.array(
            [1.0 if (i > np.max(array[0])) | (i < np.min(array[0])) else 0.0 for i in z]
        )

    return interp_rslts, extrapolation_filter


@njit()
def replacement(a, c):
    """Auxiliary function to solve the fact that numba doesn't support
    some types of advanced array sclicing.
    """
    b = c.copy()
    for counter in prange(a.shape[0]):
        for counter2 in prange(a.shape[1]):
            if a[counter, counter2] == 1:
                b[counter, counter2] = -np.inf
            else:
                continue
    return b


@njit()
def create_top_array(a):
    """Second auxiliary function to solve the situation that numba neither
    supports advanced array sclicing nor numpy ufuncs with axis reduction.
    In fact this function determines the maximal values for each column in a
    on axis=1 and returns an array of the same shape that as well as an
    boolean indictaor array.
    """
    # create container array for max values
    b = np.empty(shape=a.shape, dtype=np.float64)
    max_cont = np.empty(shape=(a.shape[1],), dtype=np.float64)

    # loop over axis 1
    for counter1 in range(a.shape[1]):
        # set all elements of b in this columns equal to the maximal value in a
        max_cont[counter1] = np.max(a[:, counter1])
    # fill b array
    for i in range(a.shape[0]):
        b[i, :] = max_cont
    return b, a == b


@njit()
def get_indices_deleted_values2(a, b):
    """This function determines the elements in a that are not in b."""
    list_ = np.empty((0,), dtype=np.int64)
    for counter, i in enumerate(a[0]):
        if i not in b:
            list_ = np.append(list_, np.array([counter]))

    return list_


@njit()
def get_indices_deleted_values(a, b):
    list_ = List.empty_list(np.int64)
    for counter, i in enumerate(a[0]):
        if i not in b:
            list_.append(counter)

    return list_


@njit()
def _intersection_function(z, sections, ln1, ln2):
    z = np.array([z])
    a = custom_interp1d(z, sections[ln2][0], sections[ln2][1])
    b = custom_interp1d(z, sections[ln1][0], sections[ln1][1])
    return (a - b)[0]


@njit()
def copy_int_jit(a):
    """This is a workaround since I don't get how i can copy integer values
    in a way that is numba conform."""
    # TODO: Check out if there is a simpler solution
    b = np.array([a], dtype=np.int64)
    return b[0]


@njit()
def split_policy_array(policy, c):
    left_indx = np.max(np.where(policy[0] < c[0])[0])
    right_indx = np.min(np.where(policy[0] >= c[0])[0])
    return left_indx, right_indx


@njit()
def insert(array1, array2, indx):
    """This function inserts an array (array1) t position
    indx in another array (array2).
    """
    return np.append(
        array2[:, :indx], np.append(array1, array2[:, indx:], axis=1), axis=1
    )


@njit()
def delete_indices(policy, removed_indices):
    """Based on a list object that contains the indices of elements that
    were removed from the grid during the upper envelope, this function
    .
    """
    # delete indices
    aux = [removed_indices[i] for i in range(len(removed_indices))]
    aux = np.array(aux)
    x = np.delete(policy[0, :], aux)
    y = np.delete(policy[1, :], aux)
    return np.append(np.expand_dims(x, axis=0), np.expand_dims(y, axis=0), axis=0)


@njit()
def update_policy(policy, new_points, removed_indices):
    """This function updates the policy rule by conducting the following steps:
    1.) Based on the application of the upper envelope on the value function it
    deletes unnecessary points in the grid
    2.) For each point added during the upper envelope step, the function interpolates
    from the left and from the right and inserts the points and the interpolated
    consumption rule into the already existing policy container.
    """
    updated_policy = policy.copy()
    # add new points to grid (and capture the discontinuity adequately)
    # delete removed points from grid
    if removed_indices.shape[0] > 0:
        updated_policy = delete_indices(updated_policy, removed_indices)
    # sort policy array
    updated_policy = updated_policy[:, updated_policy[0].argsort()]

    if new_points.shape[1] > 0:
        # analyze policy function:
        for element in new_points[0, :]:
            # for all new points interpolate new policy values (from the left and the
            # right side)
            left_indx, right_indx = split_policy_array(
                updated_policy, np.array([element])
            )

            # interpolate policy function (from both sides)
            left_interpolation = custom_interp1d(
                np.array([element]),
                updated_policy[0, left_indx : left_indx + 2],
                updated_policy[1, left_indx : left_indx + 2],
            )
            right_interpolation = custom_interp1d(
                np.array([element]),
                updated_policy[0, right_indx - 1 : right_indx + 1],
                updated_policy[1, right_indx - 1 : right_indx + 1],
            )
            # create an array based on the interpolation values

            a = np.expand_dims(
                np.array([element - 0.001 * 2.2204e-16, element]), axis=1
            ).T
            b = np.expand_dims(
                np.append(left_interpolation, right_interpolation), axis=1
            ).T
            add_array = np.append(a, b, axis=0)

            # update policy array by inserting this two values for the new point

            updated_policy = insert(add_array, updated_policy, left_indx + 1)
    return updated_policy
