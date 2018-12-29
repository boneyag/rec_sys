import pandas as pd
import numpy as np

def length_features(reports):

    df_len = []

    for item in range(len(reports)):
        dict_len = {}
        for k1, v1 in reports[item].items():
            if (k1 != 'title'):
                for k2, v2 in reports[item][k1]['text'].items():
                    dict_len[k2] = len(v2.split())
        df_len.append(pd.DataFrame.from_dict(dict_len, orient = 'index', dtype = float, columns = ['count']))

    return df_len

def slen(df_len):

    for report in df_len:
        max_len = np.max(report.loc[:,'count'])
        report.loc[:,'SLEN'] = report.loc[:,'count']/max_len

    return df_len

def slen2(df_len, structures):
    
    for report in range(len(structures)):
        for turn in range(len(structures[report])):
            start = str(turn+1)+".1"
            end = str(turn+1)+"."+str(structures[report][turn+1])
            # print(start,":",end)
            max_len = np.max(df_len[report].loc[start:end,'count'])
            df_len[report].loc[start:end,'SLEN2'] = df_len[report].loc[start:end,'count']/max_len

    return df_len