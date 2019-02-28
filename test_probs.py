import pprint

def test(sprob, tprob, r_words):

    pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(sprob[1])
    # pp.pprint(tprob[1])
    # pp.pprint(r_words[1])
    
    all_words = []
    sprob_words = 0
    tprob_words = 0

    for i in range(1, len(r_words)+1):
        
        for k in r_words[i].keys():
            for j in range(len(r_words[i][k])):
                all_words.append(r_words[i][k][j])

        # print(all_words)
        

    for k in sprob.keys():
        sprob_words += len(sprob[k].keys())

    for k in tprob.keys():
        tprob_words += len(tprob[k].keys())

    print('All words in bug reports:',len(list(set(all_words))))
    print('# words in sprob:',sprob_words)
    print('# words in tprob:',tprob_words)