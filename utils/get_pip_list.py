import pkg_resources


def get_dist(only_name: bool = False):
    return [str(d).split(' ')[0] if only_name else str(d).split(' ') for d in pkg_resources.working_set]