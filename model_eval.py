import pandas as pd
import statsmodels.api as sm
import random

def basic_train_test(f_len, summaries, word_count):
    
    report_list = [i for i in range(1, 37)]

    random.seed(2001)
    train_list = random.sample(report_list, 32)
    test_list = [i for i in report_list if i not in train_list]

    # print(report_list)
    # print(train_list)
    # print(test_list)

    for i in range(len(train_list)):
        x_train = [f_len[i] for i in train_list]
        x_train = merge_data(x_train)
        y_train = [summaries[i] for i in train_list]
        y_train = merge_data(y_train)

    print("train list",len(train_list))
    print("feature list",len(f_len))
    print("summary list",len(summaries))

def lou(features, summaries, word_count):

    for i in range(35,len(features)):
        # print(i)
        # i = 18
        w_count = word_count[i]
        x_test_c = features[i][['count']].copy()
        x_test = features[i]
        x_test.drop('count', axis=1, inplace=True)
        x_train_list = features
        x_train_list.pop(i)
        y_test = summaries[i]
        y_train_list = summaries
        y_train_list.pop(i)

        x_train = merge_data(x_train_list)
        x_train.drop('count', axis=1, inplace=True)
        y_train = merge_data(y_train_list)
        # print(x_train.shape,"\t",y_train.shape)
        # print(x_test.shape,"\t",y_test.shape)
        # print(x_train.head())
        # print(x_test_c.head())

        model = sm.OLS(y_train, x_train, missing='drop').fit()
        # print(model.summary())

        predictions = model.predict(x_test)
        predictions = predictions.sort_values(ascending=False)
        # print(predictions.sort_values(ascending=False))
        # print(type(predictions))

        print("\n\nLeave one out cross validation \nTest report:",i+1)
        match(predictions, w_count, x_test_c, y_test)
        i += 1

def merge_data(features):
    res = pd.concat([features[0],features[1]], ignore_index = True)
    for i in range(2,len(features)):
        res = pd.concat([res, features[i]], ignore_index = True)

    return res

def match(sorted, w_count, x_test_c, y_test):
    sw_count = 0
    summary = []

    true_positive = 0
    actual_summary_count = 0

    print("total word count of report:", w_count)
    print("summary word count: ~", w_count/3)

    for i,v in sorted.items():
        sw_count += x_test_c.at[i,'count']
        summary.append(i)
        # print(sw_count)
        if (sw_count >= w_count/3):
            break

    print(summary)
    for item, val in y_test['y'].iteritems():
        if(val == 1.0):
            actual_summary_count += 1
    for item in summary:
        if (y_test.at[item,'y'] == 1.0):
           true_positive += 1 

    precision = true_positive/len(summary)
    recall = true_positive/actual_summary_count
    f1 = 0
    if((precision+recall) != 0.0):
        f1 = (2*precision*recall)/(precision+recall)
    print("Precision=",precision)
    print("Recall=",recall)
    print("F1=",f1)

def reg_model(features, summaries, word_count):

    x_train_list = features
    y_train_list = summaries

    x_train = merge_data(x_train_list)
    x_train.drop('count', axis=1, inplace=True)
    y_train = merge_data(y_train_list)

    model = sm.OLS(y_train, x_train, missing='drop').fit()
    print(model.summary())