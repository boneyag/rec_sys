from itertools import combinations
from itertools import chain

def create_combinations(comp_list, num_picks):

    return list(combinations(comp_list, num_picks))

def concat_lists(arg_tuple):
    temp = []
    for e in arg_tuple:
        temp.extend(e)

    return temp