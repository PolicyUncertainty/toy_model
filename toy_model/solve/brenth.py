"""This module contains a pure just in time compiler conform python implementation of
brenth's method with hyperbolic extrapolation."""
import numpy as np
from numba import njit
from numba.typed import List


@njit(fastmath=True)
def brenth_(f, a, b, xtol, maxiter, *args):
    """This function conducts brent's method with hyperbolic extrapolation
    to find the root of f in the interval [a, b]. All credits to the developers
    of the pyroots package...
    """

    counter = 0
    # check for easy solutions
    (
        convergence,
        x_current,
        x_steps,
        fx_steps,
        f_previous,
        f_current,
    ) = check_input_values(f, a, b, xtol, *args)
    # relabel

    if convergence is False:
        x_previous, x_current = a, b
        xblk, fblk, spre, scur = 0, 0, 0, 0

    while (convergence is False) & (maxiter > counter):
        if f_previous * f_current < 0:
            xblk = x_previous
            fblk = f_previous
            spre = scur = x_current - x_previous
        if abs(fblk) < abs(f_current):
            x_previous = x_current
            x_current = xblk
            xblk = x_previous
            f_previous = f_current
            f_current = fblk
            fblk = f_previous
        sbis = (xblk - x_current) / 2
        if abs(sbis) < xtol:
            print("Warning: Bracket is smaller than tolerance.")

            return np.nan
        # check short step
        if abs(spre) > xtol and abs(f_current) < abs(f_previous):
            if x_previous == xblk:
                # interpolate
                stry = -f_current * (x_current - x_previous) / (f_current - f_previous)
            else:
                # extrapolate
                dpre = (f_previous - f_current) / (x_previous - x_current)
                dblk = (fblk - f_current) / (xblk - x_current)
                stry = extrapolate(f_current, f_previous, fblk, dpre, dblk)

            # check short step
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - xtol):
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
        else:
            # bisect
            spre = sbis
            scur = sbis

        x_previous = x_current
        f_previous = f_current
        if abs(scur) > xtol:
            x_current += scur
        else:
            x_current += xtol if (sbis > 0) else -xtol
        f_current = f(x_current, *args)  # function evaluation
        x_steps.append(x_current)
        fx_steps.append(f_current)

        if check_convergence_fast_math(0.0, f_current, xtol):
            convergence = True
        counter += 1

    return x_current


@njit(fastmath=True)
def extrapolate(fcur, fpre, fblk, dpre, dblk):
    """This function performs the hyperbolic extrapolation."""
    return -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))


@njit(fastmath=True)
def check_convergence_fast_math(a, b, xtol):
    """This function checks whether a and b fullfill the
    convergence criteria."""

    # shortcut to handle the case where both a and b are
    # actually infinity
    if a == b:
        return True

    else:
        diff = abs(a - b)
        max_ab = max(abs(a), abs(b), 1)
        if max_ab >= diff or max_ab > 1:
            return diff <= xtol  # absolute error
        else:
            return diff < xtol * max_ab  # relative  error


@njit()
def check_input_values(f, a, b, xtol, *args):
    """This function checks if one of the following cases is true
    1.) the width of the brackets is actually smaller than the
    provided tolerance level
    2.) lower bound a is the solution (abs(f(a)) < xtol)
    3.) upper bound b is the solution (abs(f(b)) < xtol)
    4.) condition for the existence of a root in the interval is
    not fulfilled ( f(a) * f(b) ) > 0.0
    """
    fx_steps = List()
    x_steps = List()

    # check if the brackert [a;b] is actually wide enough:
    if check_convergence_fast_math(a, b, xtol):
        print("Warning: Bracket is smaller than tolerance.")

        return False, np.nan, x_steps, fx_steps, np.nan, np.nan

    f_a = f(a, *args)
    f_b = f(b, *args)

    # check if lower bound is a root
    if check_convergence_fast_math(f_a, 0.0, xtol):
        return True, a, x_steps.append(a), fx_steps.append(f_a), f_a, f_b
    #
    # check if upper bound is a root
    if check_convergence_fast_math(f_b, 0.0, xtol):
        return True, b, x_steps.append(b), fx_steps.append(f_b), f_a, f_b

    # check if there can actually be a root in the bracket
    if f_a * f_b > 0.0:
        print("Warning: Root is not bracketed.")

        return False, np.nan, x_steps, fx_steps, f_a, f_b
    return False, np.nan, x_steps, fx_steps, f_a, f_b
