from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from model_eval import merge_data
import pandas as pd
import numpy as np

def test_logiReg(features, summaries, word_count, s_word_count):
    
    for i in range(len(features)):
        # print(i)
        x_train_list = features.copy()
        y_train_list = summaries.copy()

        x_test = features[i].copy()
        x_train_list.pop(i)
        x_train = merge_data(x_train_list)
        # print(x_train.head())
        # print(x_train.head())

        
        y_test = summaries[i]
        y_train_list.pop(i)
        y_train = merge_data(y_train_list)
        # print(y_train.to_numpy().ravel().shape)
        # print(type(y_train.to_numpy().ravel()))

        logi_model = LogisticRegression(solver='liblinear')

        logi_model.fit(x_train, y_train.to_numpy().ravel())

        y_prob = logi_model.predict_proba(x_test)
        # print(logi_model.predict(x_test))
        y_predict = custom_predict(y_prob, word_count[i], s_word_count[i+1])

        score = logi_model.score(x_test, y_test.to_numpy().ravel())
        precision = precision_score(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        fscore = f1_score(y_test, y_predict)

        print(i,':',score)
        print(i,':',precision)
        print(i,':',recall)
        print(i,':',fscore,'\n')

        # print(i,':',y_predict,'\n')
        # print(i,':',y_test,'\n')
        del x_train_list[:]
        del x_train_list
        del y_train_list[:]
        del y_train_list

        break

def custom_predict(y_prob, word_count, s_word_count_dict):
    
    current_word_count = 0
    summary_pred = np.zeros(len(s_word_count_dict))
    s_word_count_list = []
    summary_prob = []

    for i in y_prob:
        summary_prob.append(i[1])

    for key in s_word_count_dict.keys():
        s_word_count_list.append(s_word_count_dict[key])

    # print(word_count/4)
    while current_word_count <= word_count/4:
        max_index = summary_prob.index(max(summary_prob))
        summary_pred[max_index] = 1.0
        current_word_count += s_word_count_list[max_index]
        summary_prob[max_index] = -1

        print(current_word_count)
        if current_word_count >= word_count:
            break

    # print(summary_pred)
    # print(summary_prob)

    return summary_pred