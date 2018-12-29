import RAKE
import re

def kws(reports):

    chars_to_remove = ['?', '!', '[', ']', '`', '\'\'', '<', '>', '(', ')', ',', ':']
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'

    sentences = {}

    i = 1
    for report in reports:
        sentences[i] = []
        for k1 in report.keys():
            if (k1 != 'title'):
                for k2 in report[k1]['text'].keys():
                    sentence = report[k1]['text'][k2]
                    sentence = re.sub(r'(?<!\d)\.(?!\d)', '', sentence)
                    sentence = re.sub(rx, '', sentence)
                    sentence = sentence.lower()

                    sentences[i].append(sentence)

        Rake = RAKE.Rake(RAKE.SmartStopList())

        for s in sentences[i]:
            print(s)
            Rake.run(s)
        i += 1
        break

    return sentences