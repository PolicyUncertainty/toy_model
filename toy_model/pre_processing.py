import collections


def dict_to_namedtuple_spec(dictionary, name):
    """Coverts non-nested dictionary to namedtuple"""

    return collections.namedtuple(f"{name}_spec", dictionary.keys())(**dictionary)
