import re
import html
from nltk.tokenize import word_tokenize

def clean(reports):

    for report in reports:
        for turn in report.keys():
            if(turn != "title"):
                for key,val in report[turn]['text'].items():
                    report[turn]['text'][key] = html.unescape(val)
                    report[turn]['text'][key] = report[turn]['text'][key].replace('\\','')
                    # print(report[turn]['text'][key])
                report[turn]['user']=report[turn]['user'].replace('\'','')
                report[turn]['date']=report[turn]['date'].replace('\'','')               

    return reports