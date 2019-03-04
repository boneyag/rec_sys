import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd
import numpy as np

import pprint

pp = pprint.PrettyPrinter(indent=4)
def weight(reports):

    stop_words = set(stopwords.words('english'))

    words = {}
    s_words = {}
    t_words = {}
    r_words = {}

    # pp.pprint(reports[30])

    i = 1
    for report in reports:
        # words dict contains all the word tokens in a bug report
        # s_word dict contains word tokens by each commentor
        # t_words dict contains word tokens in each turn
        # r_words dict contains word tokens in each sentence of a comment
        words[i] = []
        s_words[i] = {}
        t_words[i] = {}
        r_words[i] = {}
        for k1 in report.keys():
            if (k1 != 'title'):
                
                user = report[k1]['user']
                t_words[i][k1] = []
                if user not in s_words[i]:
                    s_words[i][user]=[]

                for k2 in report[k1]['text'].keys():
                    sentence = report[k1]['text'][k2]
                    
                    # sentence = preprocess(sentence)
                    # if (i==8) and (k2 == '4.8'):
                    #     print(sentence)         
                    word_tokens = word_tokenize(sentence)
                    
                    r_words[i][k2] = []
                    temp = []

                    for w in word_tokens:
                        w = preprocess(w)
                        if ((w not in stop_words) and (len(w)>2)):
                            if (w not in words[i]):
                                words[i].append(w)
                            r_words[i][k2].append(w)
                            temp.append(w)

                    s_words[i][user].extend(temp)
                    t_words[i][k1].extend(temp)
        i += 1
            
        for k1 in s_words.keys():
            for k2 in s_words[k1].keys():
                s_words[k1][k2] = nltk.FreqDist(s_words[k1][k2])

        for k1 in t_words.keys():
            for k2 in t_words[k1].keys():
                t_words[k1][k2] = nltk.FreqDist(t_words[k1][k2])
    # print('all tokens in a bug report')
    # print(words[1])

    # print('\nall tokens in a bug report by user')
    # pp.pprint(s_words[1])

    # print('\nall tokens in a bug report by turn')
    # pp.pprint(t_words[1])

    # print('\nall tokens in a bug report by sentence')
    # print(r_words[1])

    sprob = {}
    tprob = {}
    for r_key in words.keys():
        sprob[r_key] = {}
        tprob[r_key] = {}
        for i in range(len(words[r_key])):
            s_max = 0
            s_sum = 0
            t_max = 0
            t_sum = 0
            for u_key in s_words[r_key].keys():
                if words[r_key][i] in s_words[r_key][u_key]:
                    if s_words[r_key][u_key][words[r_key][i]] > s_max:
                        s_max = s_words[r_key][u_key][words[r_key][i]]
                    s_sum += s_words[r_key][u_key][words[r_key][i]]
            if s_sum != 0:
                sprob[r_key][words[r_key][i]] = s_max / s_sum
            else:
                sprob[r_key][words[r_key][i]] = s_sum

            for t_key in t_words[r_key].keys():
                if words[r_key][i] in t_words[r_key][t_key]:
                    if t_words[r_key][t_key][words[r_key][i]] > t_max:
                        t_max = t_words[r_key][t_key][words[r_key][i]]
                    t_sum += t_words[r_key][t_key][words[r_key][i]]
            if t_sum != 0:
                tprob[r_key][words[r_key][i]] = t_max / t_sum
            else:
                tprob[r_key][words[r_key][i]] = t_sum
            
    return sprob, tprob, r_words       

def preprocess(sentence):
    sentence = re.sub(r'\'', '', sentence)
    sentence = re.sub(r'\"', '', sentence)
    sentence = re.sub(r'\.+$', '', sentence)
    sentence = re.sub(r'\.+\.', '', sentence)    
    sentence = re.sub(r'\?', '', sentence)
    sentence = re.sub(r'^.*\>', '', sentence)
    sentence = re.sub(r'\(*[0-9]\)', '', sentence)
    sentence = re.sub(r'..\)', '', sentence)
    sentence = re.sub(r'\(', '', sentence)
    sentence = re.sub(r'\)', '', sentence)  
    sentence = re.sub(r'\[', '', sentence)
    sentence = re.sub(r'\]', '', sentence)   
    sentence = re.sub(r'\!', '', sentence) 
    sentence = re.sub(r',', '', sentence) 
    # remove URLs
    sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
    # remove file paths with at leaset three levels
    sentence = re.sub(r'(.+\/.+\/.+)+[\r\n]*', '', sentence, flags=re.MULTILINE)
    sentence = re.sub(r'/',' ', sentence)
    sentence = sentence.lower()

    return sentence

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
                            'SMS': s_temp_sum,
                            'MXS': s_temp_max,
                            'MNS': s_temp_avg,
                            'SMT': t_temp_sum,
                            'MXT': t_temp_max,
                            'MNT': t_temp_avg
            }
        
        df_lex.append(pd.DataFrame.from_dict(dict_lex, orient = 'index', dtype = float, columns = ['SMS','MXS','MNS','SMT','MXT','MNT']))

        # print(df_lex)    
        # break

    return df_lex