def count(reports):

    word_count = []

    for report in reports:
        count = 0
        for k1 in report.keys():
            if(k1 != 'title'):
                for k2 in report[k1]['text'].keys():
                    count += len(report[k1]['text'][k2].split())
        
        word_count.append(count)

    return word_count