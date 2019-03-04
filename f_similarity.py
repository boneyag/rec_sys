from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import itertools
import pprint

pp = pprint.PrettyPrinter(indent=4)

def sentence_sim(r_words, x_prob, report_structure, num):

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(r_words[1]['1.1'])

    df_sim = []

    for i in range(len(report_structure)):
        
        dict_sim = {}
        # since dictionary keys have no order we create a list of sentence IDs to 
        # access previous and subsequent sentences
        sentence_ids = []

        for key in report_structure[i].keys():
            for k in range(1, report_structure[i][key]+1):
                id = str(str(key)+'.'+str(k))
                sentence_ids.append(id)
        # pp.pprint(x_prob[i+1])

        for j in range(len(sentence_ids)):
            # print(i+1,":",sentence_ids[j])
            if (j == 0):
                if (len(r_words[i+1][sentence_ids[j]]) != 0) and (len(r_words[i+1][sentence_ids[j+1]]) != 0):
                    vec = list(set(itertools.chain(r_words[i+1][sentence_ids[j]], r_words[i+1][sentence_ids[j+1]])))
                    
                    docs = np.ndarray(shape=(2,len(vec)))

                    for e in range(len(vec)):
                        if (vec[e] in r_words[i+1][sentence_ids[j]]):
                            # if (vec[e] in x_prob[i+1]):
                            docs[0][e] = x_prob[i+1][vec[e]]
                        else:
                            docs[0][e] = 0
                        
                        if (vec[e] in r_words[i+1][sentence_ids[j+1]]):
                            # if (vec[e] in x_prob[i+1]):
                            docs[1][e] = x_prob[i+1][vec[e]]
                        else:
                            docs[1][e] = 0

                    # cosine similarity output is an array(nx_dim, ny_dim). Therefore, to get the value only use [0,0] indexing
                    dict_sim[sentence_ids[j]] = cosine_similarity(docs[0,:].reshape(1, -1), docs[1,:].reshape(1, -1))[0][0]
                else:
                    dict_sim[sentence_ids[j]] = 0.0
      
            elif (j == len(sentence_ids)-1):
                if (len(r_words[i+1][sentence_ids[j]]) != 0) and (len(r_words[i+1][sentence_ids[j-11]]) != 0):
                    vec = list(set(itertools.chain(r_words[i+1][sentence_ids[j-1]], r_words[i+1][sentence_ids[j]])))

                    docs = np.ndarray(shape=(2,len(vec)))

                    for e in range(len(vec)):
                        if (vec[e] in r_words[i+1][sentence_ids[j-1]]):
                            # if (vec[e] in x_prob[i+1]):
                            docs[0][e] = x_prob[i+1][vec[e]]
                        else:
                            docs[0][e] = 0

                        if (vec[e] in r_words[i+1][sentence_ids[j]]):
                            # if (vec[e] in x_prob[i+1]):
                            docs[1][e] = x_prob[i+1][vec[e]]
                        else:
                            docs[1][e] = 0

                    dict_sim[sentence_ids[j]] = cosine_similarity(docs[0,:].reshape(1, -1), docs[1,:].reshape(1, -1))[0][0]
                else:
                    dict_sim[sentence_ids[j]] = 0.0

            else:
                if (len(r_words[i+1][sentence_ids[j]]) == 0):
                    dict_sim[sentence_ids[j]] = 0.0
                else:
                    if (len(r_words[i+1][sentence_ids[j-1]]) == 0):
                        vec = list(set(itertools.chain(r_words[i+1][sentence_ids[j]], r_words[i+1][sentence_ids[j+1]])))

                        docs = np.ndarray(shape=(2, len(vec)))

                        for e in range(len(vec)):
                            if (vec[e] in r_words[i+1][sentence_ids[j]]):
                                docs[0][e] = x_prob[i+1][vec[e]]
                            else:
                                docs[0][e] = 0
                            
                            if (vec[e] in r_words[i+1][sentence_ids[j+1]]):
                                docs[1][e] = x_prob[i+1][vec[e]]
                            else:
                                docs[1][e] = 0

                        # cosine similarity output is an array(nx_dim, ny_dim). Therefore, to get the value only use [0,0] indexing
                        dict_sim[sentence_ids[j]] = cosine_similarity(docs[0,:].reshape(1, -1), docs[1,:].reshape(1, -1))[0][0]

                    elif (len(r_words[i+1][sentence_ids[j+1]]) == 0):
                        vec = list(set(itertools.chain(r_words[i+1][sentence_ids[j-1]], r_words[i+1][sentence_ids[j]])))

                        docs = np.ndarray(shape=(2,len(vec)))

                        for e in range(len(vec)):
                            if (vec[e] in r_words[i+1][sentence_ids[j-1]]):
                                # if (vec[e] in x_prob[i+1]):
                                docs[0][e] = x_prob[i+1][vec[e]]
                            else:
                                docs[0][e] = 0

                            if (vec[e] in r_words[i+1][sentence_ids[j]]):
                                # if (vec[e] in x_prob[i+1]):
                                docs[1][e] = x_prob[i+1][vec[e]]
                            else:
                                docs[1][e] = 0

                        dict_sim[sentence_ids[j]] = cosine_similarity(docs[0,:].reshape(1, -1), docs[1,:].reshape(1, -1))[0][0]
                    else:
                        vec = list(set(itertools.chain(r_words[i+1][sentence_ids[j-1]], r_words[i+1][sentence_ids[j]], r_words[i+1][sentence_ids[j+1]])))
                        
                        docs = np.ndarray(shape=(3,len(vec)))

                        for e in range(len(vec)):
                            if (vec[e] in r_words[i+1][sentence_ids[j-1]]):
                                # if (vec[e] in x_prob[i+1]):
                                docs[1][e] = x_prob[i+1][vec[e]]
                            else:
                                docs[1][e] = 0

                            if (vec[e] in r_words[i+1][sentence_ids[j]]):
                                # if (vec[e] in x_prob[i+1]):
                                docs[0][e] = x_prob[i+1][vec[e]]
                            else:
                                docs[0][e] = 0

                            if (vec[e] in r_words[i+1][sentence_ids[j+1]]):
                                # if (vec[e] in x_prob[i+1]):
                                docs[2][e] = x_prob[i+1][vec[e]]
                            else:
                                docs[2][e] = 0

                        
                            
                        dict_sim[sentence_ids[j]] = (cosine_similarity(docs[0,:].reshape(1, -1), docs[1,:].reshape(1, -1))[0][0] +\
                            cosine_similarity(docs[0,:].reshape(1, -1), docs[2,:].reshape(1, -1))[0][0]) / 2
            
        # print(dict_sim)
        
        if (num == 1):
            df_sim.append(pd.DataFrame.from_dict(dict_sim, orient = 'index', dtype = float, columns = ['COS1']))
        
        if (num == 2):
            df_sim.append(pd.DataFrame.from_dict(dict_sim, orient = 'index', dtype = float, columns = ['COS2']))

        # print(df_sim[0])

        # break
    
    return df_sim