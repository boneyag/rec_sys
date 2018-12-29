import xml.etree.ElementTree as ET
import pprint
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

reports = []           #contian bug report data like senteces
structure = []         #contain bug report structure like how many turn and how many sentences in each turn

tree = ET.parse('sample_bugs.xml')
root = tree.getroot()

#reading through XML and get sentences along with user and date-time related to a # BUG
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
# pp.pprint(reports[0])



#feature extraction begins here
#lexical features

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

# pp.pprint(sprob)
# pp.pprint(tprob)


#dataframe list
df = []

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

    df.append(pd.DataFrame(p_dict))
    keys.clear()

    r += 1

for i in range(len(df)):
    print(i)
    for j in df[i].columns.values:
        if len(df[i].loc['sprob',j]) == 0:
            df[i].loc['MXS',j] = 0
            df[i].loc['SXS',j] = 0
            df[i].loc['MNS',j] = 0
        else:
            df[i].loc['MXS',j] = np.max(df[i].loc['sprob',j])
            df[i].loc['SXS',j] = np.sum(df[i].loc['sprob',j])
            df[i].loc['MNS',j] = np.mean(df[i].loc['sprob',j])
        if len(df[i].loc['tprob',j]) == 0:
            df[i].loc['MXT',j] = 0
            df[i].loc['SXT',j] = 0
            df[i].loc['MNT',j] = 0
        else:
            df[i].loc['MXT',j] = np.max(df[i].loc['tprob',j])
            df[i].loc['SXT',j] = np.sum(df[i].loc['tprob',j])
            df[i].loc['MNT',j] = np.mean(df[i].loc['tprob',j])
    df[i].drop(['sprob', 'tprob'], inplace = True)

print(df[0])
print(df[1])
# print(df[0].loc['tprob','3.1'])
# print(np.max(df[0].loc['tprob','3.1']))
# print(np.sum(df[0].loc['tprob','3.1']))
# print(np.mean(df[0].loc['tprob','3.1']))
# print(df[0].loc['MXS','3.1'])
# print(df[0].loc['SMS','3.1'])
# print(df[0].loc['MNS','3.1'])
# print(df[1])
