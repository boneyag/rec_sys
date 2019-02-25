from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
import pprint

def sentence_sim(r_words, s_prob, t_prob, report_structure):

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(r_words[1]['1.1'])

    for i in range(len(report_structure)):
        sentence_ids = []

        for key in report_structure[i].keys():
            for k in range(1, report_structure[i][key]+1):
                id = str(str(key)+'.'+str(k))
                sentence_ids.append(id)

        for j in range(len(sentence_ids)):
            if (j == 0):
                vec = list(set(itertools.chain(r_words[i+1][sentence_ids[j]], r_words[i+1][sentence_ids[j+1]])))
                 
                docs = np.ndarray(shape=(2,len(vec)))

                for e in range(len(vec)):
                    if (vec[e] in r_words[i+1][sentence_ids[j]]):
                        if (vec[e] in s_prob[i+1]):
                            docs[0][e] = s_prob[i+1][vec[e]]
                    else:
                        docs[0][e] = 0
                    
                    if (vec[e] in r_words[i+1][sentence_ids[j+1]]):
                        if (vec[e] in s_prob[i+1]):
                            docs[1][e] = s_prob[i+1][vec[e]]
                    else:
                        docs[1][e] = 0

                print(cosine_similarity(docs[0,:].reshape(1, -1), docs[1,:].reshape(1, -1)))
      
            elif (j == len(sentence_ids)-1):
                vec = list(set(itertools.chain(r_words[i+1][sentence_ids[j-1]], r_words[i+1][sentence_ids[j]])))

                docs = np.ndarray(shape=(2,len(vec)))

                for e in range(len(vec)):
                    if (vec[e] in r_words[i+1][sentence_ids[j-1]]):
                        if (vec[e] in s_prob[i+1]):
                            docs[0][e] = s_prob[i+1][vec[e]]
                    else:
                        docs[0][e] = 0

                    if (vec[e] in r_words[i+1][sentence_ids[j]]):
                        if (vec[e] in s_prob[i+1]):
                            docs[1][e] = s_prob[i+1][vec[e]]
                    else:
                        docs[1][e] = 0

                print(cosine_similarity(docs[0,:].reshape(1, -1), docs[1,:].reshape(1, -1)))

            else:
                vec = list(set(itertools.chain(r_words[i+1][sentence_ids[j-1]], r_words[i+1][sentence_ids[j]], r_words[i+1][sentence_ids[j+1]])))

                docs = np.ndarray(shape=(3,len(vec)))

                for e in range(len(vec)):
                    if (vec[e] in r_words[i+1][sentence_ids[j-1]]):
                        if (vec[e] in s_prob[i+1]):
                            docs[1][e] = s_prob[i+1][vec[e]]
                    else:
                        docs[1] = 0

                    if (vec[e] in r_words[i+1][sentence_ids[j]]):
                        if (vec[e] in s_prob[i+1]):
                            docs[0][e] = s_prob[i+1][vec[e]]
                    else:
                        docs[0][e] = 0

                    if (vec[e] in r_words[i+1][sentence_ids[j+1]]):
                        if (vec[e] in s_prob[i+1]):
                            docs[2][e] = s_prob[i+1][vec[e]]
                    else:
                        docs[2][e] = 0

                print(cosine_similarity(docs[0,:].reshape(1, -1), docs[1,:].reshape(1, -1)))
                print(cosine_similarity(docs[0,:].reshape(1, -1), docs[2,:].reshape(1, -1)))
        break
