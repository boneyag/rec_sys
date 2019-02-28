from get_xml import get_bugs
from get_xml import get_summary
from f_summary import summary
from f_length import length_features
from f_length import slen
from f_length import slen2
from f_lexical import weight
from f_lexical import lex_features

from f_similaity import sentence_sim
# from key_words import kws
from test_probs import test
from word_count import count
from model_eval import basic_train_test
from model_eval import lou
from model_eval import reg_model

from linear_model import test_liblinear

from clean_sentences import clean

import pandas as pd
import pprint


bug_xml = 'bugreports.xml'
summary_xml = 'annotation.xml'

bug_reports, report_structure = get_bugs(bug_xml)

bug_reports = clean(bug_reports)

pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(report_structure[0])
# pp.pprint(bug_reports[0])

# word_count has the word count of the bug report
# sentence_word_count has the word count of each sentence
word_count, sentence_word_count = count(bug_reports)

# pp.pprint(sentence_word_count[10])

#df_summary contains a dataframe for extractive summary (GSS) of each bug report
ext_summary = get_summary(summary_xml, report_structure)
df_summary = summary(ext_summary)

# pp.pprint(df_summary[1])
# pp.pprint(df_summary[35])

#this function creates a dataframe for each bug report 
# df_len = length_features(bug_reports)

# pp.pprint(df_len[11])

#slen and slen2 return normalized length based on word count for each sentence
#slen is the normalized length by the longest sentence in the bug report
#slen2 is the normalized length by the longest sentence of a particular turn
# df_len = slen(df_len)
# df_len = slen2(df_len, report_structure)

# for j in range(len(df_len)):
#     df_len[j].drop('count', axis = 1, inplace = True)
# pp.pprint(df_len[0])
# print(df_len[9])
# basic_train_test(df_len, ext_summary, word_count)

#weight function returns weights (sprob and tprob) and the tokenized sentences 
sprob, tprob, r_words = weight(bug_reports)
# pp.pprint(r_words[1])
# pp.pprint(sprob[1])
# df_lex = lex_features(sprob, tprob, r_words)

# test(sprob, tprob, r_words)

#cosine similarity
sentence_sim(r_words, sprob, tprob, report_structure)

#keyword extraction with RAKE
# sentences = kws(bug_reports)
# pp.pprint(sentences[1])

# merge two feature dataframes 
# df_merge = []
# df_lex2 = []
# for i in range(len(df_len)):
#     df_merge.append(pd.concat([df_len[i], df_lex[i]], axis = 1, sort=True)) 
    # df_lex2.append(pd.concat([df_len[i], df_lex[i]], axis = 1, sort=True))
    # df_lex2[i].drop(['SLEN', 'SLEN2'], axis=1, inplace = True)

# print(df_merge[34])
# print(df_lex2[0])
# print(df_len[0])
# print(df_lex[1])
# lou(df_lex2, df_summary, word_count)

# reg_model(df_merge, df_summary, word_count)
# lou(df_merge, df_summary, word_count)
# test_liblinear(df_merge, df_summary, word_count)