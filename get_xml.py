import xml.etree.ElementTree as ET
import pprint

def get_bugs(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    reports = []           #contian bug report data like senteces
    structure = []         #contain bug report structure like how many turn and how many sentences in each turn

    for report in root:
        b_dict = {}
        s_dict = {}
        for item in report.iter('BugReport'):
            for title in item.iter('Title'):
                b_dict['title'] = title.text
            i = 1
            for turn in item.iter('Turn'):
                temp = {}
                for date in turn.iter('Date'):
                    temp['date'] = date.text
                for user in turn.iter('From'):
                    temp['user'] = user.text
                for text in turn.iter('Text'):
                    temp2 = {}
                    j = 1
                    for sentence in text.iter('Sentence'):
                        if(len(sentence.text)==0):
                            temp2[sentence.get('ID')] = ''
                        else:
                            temp2[sentence.get('ID')] = sentence.text
                        j += 1
                    temp['text'] = temp2
                b_dict[i] = temp
                s_dict[i] = j-1
                i += 1
            # s_dict['turns'] = i-1
        reports.append(b_dict)
        structure.append(s_dict)

    return reports, structure

def get_summary(xml_file, structure):
    # pp = pprint.PrettyPrinter(indent=4)
    summary_reports = []

    i = 1
    for report in structure:
        sum_dict = {}
        for key, val in report.items():
            for j in range(val):
                index = str(key)+"."+str(j+1)
                # index = str(i)+"."+str(key)+"."+str(j+1)
                # print(index)
                sum_dict[index] = 0
        summary_reports.append(sum_dict)
        i += 1

    # pp.pprint(summary_reports[35])

    tree = ET.parse(xml_file)
    root = tree.getroot()

    i = 1
    for report in root:
        for item in report.iter('BugReport'):
            for annotation in item.iter('Annotation'):
                for summary in annotation.iter('ExtractiveSummary'):
                    for sentence in summary.iter('Sentence'):
                        index = str(sentence.get('ID')).strip()
                        summary_reports[i-1][index] += 1
        i += 1

    # pp.pprint(summary_reports[35])

    for sr in summary_reports:
        for key, val in sr.items():
            if (val >= 2):
                sr[key] = 1
            else:
                sr[key] = 0
    
    # pp.pprint(summary_reports[35])

    return summary_reports