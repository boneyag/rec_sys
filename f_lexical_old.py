import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd
import numpy as np

import pprint

def weight(reports):
    
    stop_words = set(stopwords.words('english'))

    chars_to_remove = ['?', '!', '[', ']', '`', '\'\'', '<', '>', '(', ')', ',', ':']
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'

    words = {}
    i = 1
    for report in reports:
        words[i] = []
        for k1 in report.keys():
            if (k1 != 'title'):
                for k2 in report[k1]['text'].keys():
                    sentence = report[k1]['text'][k2]
                    sentence = re.sub(r'(?<!\d)\.(?!\d)', '', sentence)
                    sentence = re.sub(rx, '', sentence)
                    sentence = sentence.lower()
                    
                    word_tokens = word_tokenize(sentence)
                    for w in word_tokens:
                        if w not in stop_words:
                            words[i].append(w)
        
        # print(words[i])
        words[i] = list(filter(None, words[i]))
        words[i] = list(set(words[i]))
        # print(words[i])
        i += 1
        # break

    s_words = {}
    t_words = {}
    r_words = {}
    i = 1
    for report in reports:
        s_words[i] = {}
        t_words[i] = {}
        r_words[i] = {}
        for k1 in report.keys():
            if (k1 != 'title'):
                user = report[k1]['user']
                t_words[i][k1] = []
                if user not in s_words[i]:
                    s_words[i][user]=[]
        
                for k2, v in report[k1]['text'].items():
                    sentence = v
                    sentence = re.sub(r'(?<!\d)\.(?!\d)', '', sentence)
                    sentence = re.sub(rx, '', sentence)
                    sentence = sentence.lower()

                    word_tokens = word_tokenize(sentence)
                    
                    r_words[i][k2] = []
                    temp = []
                    for w in word_tokens:
                        if w not in stop_words:
                            r_words[i][k2].append(w)
                            temp.append(w)

                    s_words[i][user].extend(temp)
                    t_words[i][k1].extend(temp)
        
        i += 1
        # break

    for k1 in s_words.keys():
        for k2 in s_words[k1].keys():
            s_words[k1][k2] = nltk.FreqDist(s_words[k1][k2])

    for k1 in t_words.keys():
        for k2 in t_words[k1].keys():
            t_words[k1][k2] = nltk.FreqDist(t_words[k1][k2])

    pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(t_words[1])
    # pp.pprint(s_words[1])
    # pp.pprint(r_words[1])

    sprob = {}
    tprob = {}
    for r_key in words.keys():
        sprob[r_key] = {}
        for i in range(len(words[r_key])-1):
            max = 0
            sum = 0
            for u_key in s_words[r_key].keys():
                if words[r_key][i] in s_words[r_key][u_key]:
                    if s_words[r_key][u_key][words[r_key][i]] > max:
                        max = s_words[r_key][u_key][words[r_key][i]]
                    sum += s_words[r_key][u_key][words[r_key][i]]
            if sum != 0:
                sprob[r_key][words[r_key][i]] = max / sum
            else:
                sprob[r_key][words[r_key][i]] = sum

        tprob[r_key] = {}
        for i in range(len(words[r_key])-1):
            max = 0
            sum = 0
            for t_key in t_words[r_key].keys():
                if words[r_key][i] in t_words[r_key][t_key]:
                    if t_words[r_key][t_key][words[r_key][i]] > max:
                        max = t_words[r_key][t_key][words[r_key][i]]
                    sum += t_words[r_key][t_key][words[r_key][i]]
            if sum != 0:
                tprob[r_key][words[r_key][i]] = max / sum
            else:
                tprob[r_key][words[r_key][i]] = sum

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(tprob[35])
    # pp.pprint(sprob[35])

    return sprob, tprob, r_words

def lex_features(sprob, tprob, r_words):

    df_lex = []

    for k1 in r_words.keys():
        dict_lex = {}
        for k2 in r_words[k1].keys():
            # print(r_words[k1][k2])
            s_temp_sum = 0.0
            s_temp_max = 0.0
            s_temp_avg = 0.0
            t_temp_sum = 0.0
            t_temp_max = 0.0
            t_temp_avg = 0.0
            for word in r_words[k1][k2]:
                if word in sprob[k1]:
                    s_temp_sum += sprob[k1][word]
                    if sprob[k1][word] > s_temp_max:
                        s_temp_max = sprob[k1][word]
            if (len(r_words[k1][k2]) != 0):
                s_temp_avg = s_temp_sum / len(r_words[k1][k2])
            else:
                s_temp_avg = 0.0

            for word in r_words[k1][k2]:
                if word in tprob[k1]:
                    t_temp_sum += tprob[k1][word]
                    if tprob[k1][word] > t_temp_max:
                        t_temp_max = tprob[k1][word]
            if (len(r_words[k1][k2]) != 0):
                t_temp_avg = t_temp_sum / len(r_words[k1][k2])
            else:
                t_temp_avg = 0.0

            dict_lex[k2] = {
                            'SSM': s_temp_sum,
                            'SMX': s_temp_max,
                            'SMN': s_temp_avg,
                            'TSM': t_temp_sum,
                            'TMX': t_temp_max,
                            'TMN': t_temp_avg
            }
        
        df_lex.append(pd.DataFrame.from_dict(dict_lex, orient = 'index', dtype = float, columns = ['SSM','SMX','SMN','TSM','TMX','TMN']))

        # print(df_lex)    
        # break

    return df_lex