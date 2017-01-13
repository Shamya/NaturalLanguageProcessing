from __future__ import  division
from structperc import read_tagging_file 

#load dev data
tweet_lst = read_tagging_file('oct27.dev.txt')

#the most common tag 
pos_vocab = {}
#build vocabulary
for tweet in tweet_lst:
    pos_lst = tweet[1]
    for pos in pos_lst:
        if pos not in pos_vocab:
            pos_vocab[pos] = 1
        else:
            pos_vocab[pos] += 1

max_pos = max(pos_vocab, key=pos_vocab.get)
print "most common tag:", max_pos, 'occurs', pos_vocab[max_pos], 'times'  

#accuracy if we predict it for all tags
count = sum(pos_vocab.values())
print "accuracy if we predict it for all tags:", (pos_vocab[max_pos]/count)*100, '%'