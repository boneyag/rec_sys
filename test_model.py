import xml.etree.ElementTree as ET
import pprint
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline

import statsmodels.api as sm


reports = []           #contian bug report data like senteces
structure = []         #contain bug report structure like how many turn and how many sentences in each turn

# xml_parser = ET.XMLParser(encoding="utf-8")
tree = ET.parse('bugs.xml')
root = tree.getroot()

#reading through XML and get sentences along with user and date-time related to a #bug
#each bug report transformed to a dictionary and stored in reports vector
for report in root:
    dict = {}
    s_dict = {}
    for item in report.iter('BugReport'):
        for title in item.iter('Title'):
            # print(title.text)
            dict['title'] = title.text
        i = 1
        for turn in item.iter('Turn'):
            temp = {}
            for date in turn.iter('Date'):
                # print(date.text)
                temp['date'] = date.text
            for user in turn.iter('From'):
                # print(user.text)
                temp['user'] = user.text
            for text in turn.iter('Text'):
                temp2 = {}
                j = 1
                for sentence in text.iter('Sentence'):
                    # print(str(sentence.attrib)+":"+sentence.text)
                    temp2[sentence.get('ID')] = sentence.text
                    j += 1
                temp['text'] = temp2
            dict[i] = temp
            s_dict[i] = j-1
            i += 1
        s_dict['turns'] = i-1
    reports.append(dict)
    structure.append(s_dict)

pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(structure)
# pp.pprint(reports)


#feature extraction begins here
#lexical features sprob and tprob
words = {}

i = 1
for report in reports:
    keys = []
    for key in report.keys():
        if isinstance(key, int):
            keys.append(key)
    words[i] = []
    for key in keys:
        for key, val in report[key]['text'].items():
            words[i].append(val)
    keys.clear()

    stop_words = set(stopwords.words('english'))

    filtered_sentence = []
    for word in words[i]:
        sentence = word
        word_tokens = word_tokenize(sentence)
        # filtered_sentence.extend([w for w in word_tokens if not w in stop_words])
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)


    filtered_sentence = [re.sub('[^a-zA-Z0-9]+', '', _) for _ in filtered_sentence] #filtering symbols
    filtered_sentence = [re.sub('\d+', '', _) for _ in filtered_sentence] #filtering words with numbers
    filtered_sentence = [re.sub(r'\b\w{1,1}\b', '', _) for _ in filtered_sentence] #filtering words length 1
    filtered_sentence = list(filter(None, filtered_sentence)) #filtering empty strings
    unique_words = set(filtered_sentence)
    words[i] = list(unique_words)
    i += 1

# pp.pprint(words)

# extract all sentences posted by each user
s_words = {}
t_words = {}
i = 1
for report in reports:
    s_words[i] = {}
    t_words[i] = {}
    keys = []
    for key in report.keys():
        if isinstance(key, int):
            keys.append(key)
    for key in keys:
        t_words[i][key] = []
        user = report[key]['user']
        if user not in s_words[i]:
            s_words[i][user]=[]
        for k, v in report[key]['text'].items():
            s_words[i][user].append(v)
            t_words[i][key].append(v)
    keys.clear()
    i += 1
# pp.pprint(t_words)

for r_key in s_words.keys():
    for key in s_words[r_key].keys():
        sprob_words = []
        for sentence in s_words[r_key][key]:
            tokens = word_tokenize(sentence)
            for w in tokens:
                if w not in stop_words:
                    sprob_words.append(w)

        sprob_words = [re.sub('[^a-zA-Z0-9]+', '', _) for _ in sprob_words] #filtering symbols
        sprob_words = [re.sub('\d+', '', _) for _ in sprob_words] #filtering words with numbers
        sprob_words = [re.sub(r'\b\w{1,1}\b', '', _) for _ in sprob_words] #filtering words length 1
        sprob_words = list(filter(None, sprob_words)) #filtering empty strings

        s_words[r_key][key] = sprob_words
        s_words[r_key][key] = nltk.FreqDist(s_words[r_key][key])

# pp.pprint(t_words)

for r_key in t_words.keys():
    tprob_words = []
    for key in t_words[r_key].keys():
        for sentence in t_words[r_key][key]:
            tokens = word_tokenize(sentence)
            for w in tokens:
                if w not in stop_words:
                    tprob_words.append(w)

        tprob_words = [re.sub('[^a-zA-Z0-9]+', '', _) for _ in tprob_words] #filtering symbols
        tprob_words = [re.sub('\d+', '', _) for _ in tprob_words] #filtering words with numbers
        tprob_words = [re.sub(r'\b\w{1,1}\b', '', _) for _ in tprob_words] #filtering words length 1
        tprob_words = list(filter(None, tprob_words)) #filtering empty strings

        t_words[r_key][key] = tprob_words
        t_words[r_key][key] = nltk.FreqDist(t_words[r_key][key])

# pp.pprint(t_words)

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


#length features

#dataframe list
df_len = []

r = 0
for report in reports:
    keys = []
    for key in report.keys():
        if isinstance(key, int):
            keys.append(key)
    p_dict = {}
    for key in keys:
        for key, val in report[key]['text'].items():
            p_dict[key] = len(val.split())
    df_len.append(pd.DataFrame(p_dict, index = ['w_count',]))
    keys.clear()

    df_len[r].loc['SLEN',:] = df_len[r].loc['w_count',:].div(np.max(df_len[r].loc['w_count',]), axis = 0)
    mean = df_len[r].loc['SLEN',:].mean()
    sd = df_len[r].loc['SLEN',:].std()
    df_len[r].loc['SLEN',:] = df_len[r].loc['SLEN',:]-mean
    df_len[r].loc['SLEN',:] = df_len[r].loc['SLEN',:]/sd

    for i in range(structure[r]['turns']):
        t_strat = str(i+1)+".1"
        t_end = str(i+1)+"."+str(structure[r][i+1])
        df_len[r].loc['SLEN2',t_strat:t_end] = df_len[r].loc['w_count',t_strat:t_end].div(np.max(df_len[r].loc['w_count',t_strat:t_end]), axis = 0)
    mean = df_len[r].loc['SLEN2',:].mean()
    sd = df_len[r].loc['SLEN2',:].std()
    df_len[r].loc['SLEN2',:] = df_len[r].loc['SLEN2',:]-mean
    df_len[r].loc['SLEN2',:] = df_len[r].loc['SLEN2',:]/sd

    df_len[r].drop(['w_count'], inplace = True)
    r += 1

df_lex = []
r = 0
for report in reports:
    keys = []
    for key in report.keys():
        if isinstance(key, int):
            keys.append(key)
    p_dict = {}
    for key in keys:
        for key, val in report[key]['text'].items():
            sentence = []
            tokens = word_tokenize(val)
            for w in tokens:
                if w not in stop_words:
                    sentence.append(w)

            sentence = [re.sub('[^a-zA-Z0-9]+', '', _) for _ in sentence] #filtering symbols
            sentence = [re.sub('\d+', '', _) for _ in sentence] #filtering words with numbers
            sentence = [re.sub(r'\b\w{1,1}\b', '', _) for _ in sentence] #filtering words length 1
            sentence = list(filter(None, sentence)) #filtering empty strings

            p_dict[key] = {}
            p_dict[key]['sprob'] = []
            p_dict[key]['tprob'] = []
            for word in sentence:
                if word in sprob[r+1]:
                    p_dict[key]['sprob'].append(sprob[r+1][word])
                if word in tprob[r+1]:
                    p_dict[key]['tprob'].append(tprob[r+1][word])

    df_lex.append( pd.DataFrame(p_dict))
    keys.clear()

    r += 1

for i in range(len(df_lex)):
    for j in df_lex[i].columns.values:
        if len(df_lex[i].loc['sprob',j]) == 0:
            df_lex[i].loc['MXS',j] = 0
            df_lex[i].loc['SXS',j] = 0
            df_lex[i].loc['MNS',j] = 0
        else:
            df_lex[i].loc['MXS',j] = np.max(df_lex[i].loc['sprob',j])
            df_lex[i].loc['SXS',j] = np.sum(df_lex[i].loc['sprob',j])
            df_lex[i].loc['MNS',j] = np.mean(df_lex[i].loc['sprob',j])
        if len(df_lex[i].loc['tprob',j]) == 0:
            df_lex[i].loc['MXT',j] = 0
            df_lex[i].loc['SXT',j] = 0
            df_lex[i].loc['MNT',j] = 0
        else:
            df_lex[i].loc['MXT',j] = np.max(df_lex[i].loc['tprob',j])
            df_lex[i].loc['SXT',j] = np.sum(df_lex[i].loc['tprob',j])
            df_lex[i].loc['MNT',j] = np.mean(df_lex[i].loc['tprob',j])

    mean = df_lex[i].loc['SXS',:].mean()
    sd = df_lex[i].loc['SXS',:].std()
    df_lex[i].loc['SXS',:] = df_lex[i].loc['SXS',:]-mean
    df_lex[i].loc['SXS',:] = df_lex[i].loc['SXS',:]/sd
    mean = df_lex[i].loc['SXT',:].mean()
    sd = df_lex[i].loc['SXT',:].std()
    df_lex[i].loc['SXT',:] = df_lex[i].loc['SXT',:]-mean
    df_lex[i].loc['SXT',:] = df_lex[i].loc['SXT',:]/sd

    print(df_lex[i].loc['MXS',:].min(),":",df_lex[i].loc['MXS',:].max())
    print(df_lex[i].loc['SXS',:].min(),":",df_lex[i].loc['SXS',:].max())
    print(df_lex[i].loc['MNS',:].min(),":",df_lex[i].loc['MNS',:].max())
    print(df_lex[i].loc['MXT',:].min(),":",df_lex[i].loc['MXT',:].max())
    print(df_lex[i].loc['SXT',:].min(),":",df_lex[i].loc['SXT',:].max())
    print(df_lex[i].loc['MNT',:].min(),":",df_lex[i].loc['MNT',:].max())

    df_lex[i].drop(['sprob', 'tprob'], inplace = True)


# for i in range(len(df_lex)):
#     df_lex[i].replace([np.inf, -np.inf, np.nan], 0.0001)

# print(df_lex[0].shape)
# print(df_lex[1].shape)

# reading extractive summary
summary_reports = []

# create initial dataframe based on bug reports
stree = ET.parse('bugs.xml')
sroot = stree.getroot()

for report in sroot:
    dict = {}
    for item in report.iter('BugReport'):
        for turn in item.iter('Turn'):
            for text in turn.iter('Text'):
                for sentence in text.iter('Sentence'):
                    dict[sentence.get('ID')] = 0
    summary_reports.append(dict)

summary_df = []

for report in summary_reports:
    summary_df.append(pd.DataFrame(report, index = ['count',]))


# read extractive summary sentences and store it is summary or not
tree2 = ET.parse('bug_summary.xml')
root2 = tree2.getroot()

i = 0
for report in root2:
    for item in report.iter('BugReport'):
        for annotation in item.iter('Annotation'):
            for summary in annotation.iter('ExtractiveSummary'):
                for sentence in summary.iter('Sentence'):
                    index = str(sentence.get('ID')).strip()
                    summary_df[i].at['count',index] += 1
    i += 1

for j in range(len(summary_df)):
    summary_df[j].loc['y',:] = np.where(summary_df[j].loc['count',:] >= 2, 1, 0)
    summary_df[j].drop(['count'], inplace = True)

# print(summary_df[0])
# print(summary_df[1])

# new_df0 = pd.concat([df_lex[0], summary_df[0]], axis = 0, join = 'inner')
# new_df1 = pd.concat([df_lex[1], summary_df[1]], axis = 0, join = 'inner')

new_df = []
for j in range(len(summary_df)):
    new_df.append(pd.concat([df_len[j], df_lex[j], summary_df[j]], axis = 0, join = 'inner'))

# new_df0 = pd.concat([df_len[0], df_lex[0], summary_df[0]], axis = 0, join = 'inner')
# new_df1 = pd.concat([df_len[1], df_lex[1], summary_df[1]], axis = 0, join = 'inner')

data = pd.concat([dfn for dfn in new_df], axis = 1)

print(data.shape)
X = data.values[0:8,:]
Y = data.values[8,:]
# Y = Y.reshape(1,2360)
X = X.T
Y = Y.T
X = X.astype(float)
Y = Y.astype(float)
# X = np.around(X, decimals=4)
print(X.shape)
print(Y.shape)
# print(Y[0:15,:])

# model = sm.OLS(Y, X, missing='drop').fit()
# predictions = model.predict(X)
# print(model.summary())

#keeping for later work

# Alpha (regularization strength) of LASSO regression
# lasso_eps = 0.0001
# lasso_nalpha=20
# lasso_iter=5000
# lasso_tol=0.5
# # Min and max degree of polynomials features to consider
# degree_min = 2
# degree_max = 8
# # Test/train split
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# # Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
# for degree in range(degree_min,degree_max+1):
#     model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,tol=lasso_tol,max_iter=lasso_iter,
# normalize=True,cv=5))
#     model.fit(X_train,y_train)
#     test_pred = np.array(model.predict(X_test))
#     RMSE=np.sqrt(np.sum(np.square(test_pred-y_test)))
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test,y_test)
#     print(train_score,"\n")
#     print(test_score,"\n")
