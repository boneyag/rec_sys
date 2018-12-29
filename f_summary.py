import pandas as pd
import numpy as np

def summary(ext_summary):

    df_summary = []

    for item in range(len(ext_summary)):
        dict_sum = {}
        for k1, v1 in ext_summary[item].items():
            dict_sum[k1] = v1
        df_summary.append(pd.DataFrame.from_dict(dict_sum, orient = 'index', dtype = float, columns = ['y']))

    return df_summary