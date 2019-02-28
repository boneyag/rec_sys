from sklearn.linear_model import LogisticRegression
from model_eval import merge_data
import pandas as pd

def test_liblinear(features, summaries, word_count):

    
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

        logi_model = LogisticRegression(solver='lbfgs')

        logi_model.fit(x_train, y_train.to_numpy().ravel())

        # predictions = logi_model.predict(x_test)
        score = logi_model.score(x_test, y_test.to_numpy().ravel())

        print(i,':',score)

        del x_train_list[:]
        del x_train_list
        del y_train_list[:]
        del y_train_list

        # break