def count(reports):

    word_count = []
    sentence_word_count = {}

    i = 1
    for report in reports:
        count = 0
        sentence_word_count[i] = {}

        for k1 in report.keys():
            if(k1 != 'title'):
                for k2 in report[k1]['text'].keys():
                    sentence_word_count[i][k2] = len(report[k1]['text'][k2].split())
                    count += len(report[k1]['text'][k2].split())
        
        word_count.append(count)
        i += 1

    return word_count, sentence_word_count