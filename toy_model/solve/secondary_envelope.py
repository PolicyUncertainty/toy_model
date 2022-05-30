import numpy as np
from numba import njit

from toy_model.model_code import util
from toy_model.shared.shared_auxiliary import custom_interp1d
from toy_model.shared.shared_auxiliary import filter_array
from toy_model.solve.brenth import brenth_
from toy_model.solve.secondary_envelop_auxiliary import _intersection_function
from toy_model.solve.secondary_envelop_auxiliary import chop
from toy_model.solve.secondary_envelop_auxiliary import copy_int_jit
from toy_model.solve.secondary_envelop_auxiliary import create_top_array
from toy_model.solve.secondary_envelop_auxiliary import get_indices_deleted_values2
from toy_model.solve.secondary_envelop_auxiliary import insert
from toy_model.solve.secondary_envelop_auxiliary import interpolation_upper_envelop
from toy_model.solve.secondary_envelop_auxiliary import replacement
from toy_model.solve.secondary_envelop_auxiliary import update_policy


@njit()
def secondary_envelope_wrapper(
    value, policy, choice, model_spec_params, utility_params
):
    # check whether we have a bound back region in the constraint region
    value_ = filter_array(value)
    policy_ = filter_array(policy)
    new_value, new_policy = check_and_correct_constraint_region(
        value_, policy_, utility_params, choice, model_spec_params[3]
    )

    # conduct secondary envelope
    new_value, removed_indices, newdots = secondary_envelope(new_value)
    # adjust value and policy array if the upper envelope adjusted the value array:
    new_policy = update_policy(new_policy, newdots, removed_indices)

    return new_value, new_policy


@njit()
def secondary_envelope(values):
    """This function conducts the secondary envelope sub algorithm to
    fix the primary and secondary kinks in the value and policy function.
    This discontinuities exist because the value function is no longer concave
    (discrete decision leads to a kink at the point where the individual switches
    between )
    )
    """

    # create a copy of the value function
    copy_val = values
    # find discontinuity points ( points where M_j > M_j+1) and chop array
    cut_points, sections = chop(copy_val, repeat=True)

    # conduct upper envelope if grid is split into more than one section
    if len(sections) > 1:
        # sort the arrays within sections list
        sections = [i[:, i[0].argsort()] for i in sections]
        # determine unique x values
        unique_x_vals = np.unique(copy_val[0])
        # conduct upper envelope
        rslts, newdots = upper_envelop(sections, unique_x_vals, True, True)
        # determine removed values =
        removed_indices = get_indices_deleted_values2(copy_val, rslts)
    else:
        # return values
        rslts = values
        # set other outputs to empty arrays
        removed_indices = np.empty((0,), dtype=np.int64)
        newdots = np.empty((2, 1), dtype=np.float64)
    return rslts, removed_indices, newdots[:, 1:]


@njit()
def upper_envelop(chopped_array, unique_x_vals, full_interval, add_intersection):
    """This function conducts an upper envelope over the previously
    determined section of the value function.
    """
    # check_chopped_parts

    # chopped_array = check_chopped_parts(chopped_array)

    # for all sections interpolate over all unique x values
    interpolation_rslts, extrapolated = interpolation_upper_envelop(
        unique_x_vals, chopped_array
    )
    # adjustments
    interpolation_rslts, unique_x_vals, n = update_extrapolation_interval(
        extrapolated, unique_x_vals, interpolation_rslts, full_interval
    )

    # conduct actual envelope
    max_interpolated, top = create_top_array(interpolation_rslts)
    upper_envelop_values, intersection_points = conduct_envelope(
        unique_x_vals,
        max_interpolated,
        top,
        n,
        chopped_array,
        add_intersection,
    )
    return upper_envelop_values, intersection_points


@njit()
def conduct_envelope(
    unique_grid,
    max_interpolated,
    top,
    n,
    sections,
    add_intersection,
):
    """This function conducts the upper envelope."""

    # set up containers for the results
    solution_container = np.array([[unique_grid[0]], [max_interpolated[0, 0]]])

    # container for the intersection points
    intersection = np.empty((2, 1))

    k0 = np.where(top[:, 0] == True)[0][0]  # noqa

    for i in range(1, n):
        k1 = np.where(top[:, i] == True)[0][0]  # noqa
        # if new maximal point lays on a different chopped section
        if k1 != k0:
            # store k0 and k1 in two new variables
            ln1, ln2 = copy_int_jit(k0), copy_int_jit(k1)
            # obtain the associated grid points
            gp1, gp2 = unique_grid[i - 1], unique_grid[i]
            # combine grid points in an array
            z = np.array([gp1, gp2])
            # conduct interpolation for grid points with the help of
            # both sections k0 and k1
            y, extra_ = interpolation_upper_envelop(z, [sections[ln1], sections[ln2]])

            # if non of the points was extrapolated and none of the points
            # is actualle the cutoff (difference equal to zero)
            if (np.all(extra_ == 0)) & (np.all(np.abs(y[0] - y[1]) > 0.0)):
                while_loop_operator = False
                while while_loop_operator is False:
                    # search for intersection point of the lines via brent's method
                    new_point = brenth_(
                        _intersection_function, gp1, gp2, 1e-10, 100, sections, ln1, ln2
                    )
                    # since the new point is the intersection of both lines it doesn't
                    # matter which section we use for interpolation to obtain the
                    # value function value (...)
                    new_function_val = custom_interp1d(
                        np.array([new_point]), sections[ln1][0], sections[ln1][1]
                    )[0]

                    # determine function value at the new point for all sections
                    new_values, extra3 = interpolation_upper_envelop(
                        np.array([new_point]), sections
                    )
                    # replace values for all sections for which the value
                    #  was obtained via extrapolation
                    new_function_values = replacement(extra3, new_values)
                    # determine for which section the value is maximal

                    max_cont2, top2 = create_top_array(new_function_values)
                    ln3 = np.where(new_function_values == max_cont2)[0][0]
                    # if the maximal line is equal to k1 or k0
                    if (ln3 == ln1) | (ln3 == ln2):
                        # no need to search further
                        # add the intersection point
                        solution_container = np.append(
                            solution_container,
                            np.array([[new_point], [new_function_val]]),
                            axis=1,
                        )
                        # if add_intersection is True, the function creates
                        # a second output array that contains the intersection
                        # points

                        if add_intersection:
                            intersection = np.append(
                                intersection,
                                np.array([[new_point], [new_function_val]]),
                                axis=1,
                            )

                        if ln2 == k1:
                            while_loop_operator = True

                        else:
                            ln1 = ln2
                            gp1 = new_point
                            ln2 = k1
                            gp2 = unique_grid[i]
                    else:
                        ln2 = ln3
                        gp2 = new_point
        # add point
        if (
            np.any(np.abs(sections[k1][0] - unique_grid[i]) < 2.2204e-16)
            == True  # noqa
        ):
            solution_container = np.append(
                solution_container,
                np.array([[unique_grid[i]], [max_interpolated[0, i]]]),
                axis=1,
            )
        # update k0
        k0 = k1

    return solution_container, intersection


@njit()
def update_extrapolation_interval(
    extrapolated, unique_x_vals, interpolation_rslts, full_interval
):
    """This function determines the unique points of the sorted grid
    that are used fot the construction of the upper envelope.

    There are two different ways how this can be conducted:

    if full_interval is equal to True interpolation result points are< only
    disregarded where particular lines are extrapolated

    if full_interval is equal to False, points for which at least one
    of the interpolation is extrapolated are not used for all sections.
    """
    if not full_interval:
        # determine if a point is extrapolated for at least one section
        mask = np.sum(extrapolated, axis=0)
        # only keep interpolation points for which rslts are interpolated
        #  for all sections
        updated_interpolation_rslts = interpolation_rslts[:, mask == 0]
        # reduce unique sorte grid to this points
        unique_x_vals = unique_x_vals[mask == 0]
        # determine how many of this kind of points there are
        n = np.sum(mask == 0)
    else:
        updated_interpolation_rslts = replacement(extrapolated, interpolation_rslts)
        n = unique_x_vals.shape[0]
    return updated_interpolation_rslts, unique_x_vals, n


@njit()
def check_and_correct_constraint_region(
    value, policy, utility_params, choice, num_grid
):
    """This function checks whether the grid bends back in the constraint region
    (non convex region coincides with the credit constraint). If it does we add
    some more points in front of the grid ( note that the value function is
    deterministic in this region (end of period assets = 0).
    """
    # no bend back region in the constraint region
    new_value = value.copy()
    new_policy = policy.copy()
    minx = np.min(new_value[0, 1:])
    if new_value[0, 1] <= minx:
        return new_value, new_policy
    # if we have a bend back region (non convex) region in the constraint region
    # note that the value function is analytical between entry 0 and 1.
    else:
        # create new points between the minimal value in the grid and the first
        # one (that is not zero!!)
        new_points = np.linspace(minx, new_value[0, 1], int(num_grid / 10))[:-1]

        # calculate associated value functions
        new_values = util(new_points, utility_params, choice)

        # adjust value and policy arrays accordingly
        new_value = insert(np.vstack((new_points, new_values)), new_value, 1)

        new_policy = insert(np.vstack((new_points, new_points)), new_policy, 1)

        return new_value, new_policy
