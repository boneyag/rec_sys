import xml.etree.ElementTree as ET
import pprint
import pandas as pd
import numpy as np

summary_reports = []

stree = ET.parse('sample_bugs.xml')
sroot = stree.getroot()

for report in sroot:
    dict = {}
    i = 1
    for item in report.iter('BugReport'):
        for turn in item.iter('Turn'):
            for text in turn.iter('Text'):
                for sentence in text.iter('Sentence'):
                    dict[sentence.get('ID')] = 0
    summary_reports.append(dict)

summary_df = []

for report in summary_reports:
    summary_df.append(pd.DataFrame(report, index = ['count',]))


tree2 = ET.parse('summary_sample.xml')
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

summary_df[0].loc['y',:] = np.where(summary_df[0].loc['count',:] >= 2, 1, 0)
summary_df[1].loc['y',:] = np.where(summary_df[1].loc['count',:] >= 2, 1, 0)

print(summary_df[0])
print(summary_df[1])
