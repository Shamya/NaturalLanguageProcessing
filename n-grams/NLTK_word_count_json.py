"""
this was the code used to create the ngram count tables.
it is NOT needed to do the assignment.
"""

from __future__ import division
from collections import defaultdict
from collections import Counter
import os
import re
import nltk
import json

def load_files(PATH_TO_DATA):
    input_file_list=os.listdir(PATH_TO_DATA)

    sub_sample_rate=0.1
    fname2content={}
    for f in input_file_list:
        with open(os.path.join(PATH_TO_DATA,f),'r') as doc:
            content = doc.read()
            fname2content[f]=content
    return fname2content

filter_tokens_set=set(["br","/",">","<"])

def remove_non_ASCII(content):
    content_printable_list=[c for c in content if (32 <= ord(c) and ord(c) <= 126)]
    return ''.join(content_printable_list)


def collect_all_sentences(fname2content,sentence_splitter):
    
    all_sentence_list=[]

    for filename,content in fname2content.items():
        content_printable=remove_non_ASCII(content)
        sentences_raw = sentence_splitter.sentences_from_text(content_printable)
        sentences_toks_origcase = [nltk.word_tokenize(sent_text) for sent_text in sentences_raw]
        sentences_toks = [[w.lower() for w in sent_toks if w not in filter_tokens_set] for sent_toks in sentences_toks_origcase]
        all_sentence_list+=sentences_toks

    return all_sentence_list


def make_ngrams(tokens, ngram_size):
    """Return a list of ngrams, of given size, from the input list of tokens.
    Also include **START** and **END** tokens appropriately."""
    ngrams = []
    tokens = ['**START**'] * (ngram_size-1) + tokens + ['**END**'] * (ngram_size-1)
    for i in range(ngram_size, len(tokens)+1):
        ngrams.append( tuple(tokens[i-ngram_size:i]))
    
    return ngrams

class NgramModelCounts:
    def __init__(self):
        self.vocabulary = set()
        self.ngram_size = None
        self.ngram_counts = defaultdict(lambda:defaultdict(int))

def get_ngram_counts(sentences, ngram_size):
    """'Train' a fixed-order ngram model by doing the necessary ngram counts.
    Return a data structure that represents the counts."""
    model = NgramModelCounts()
    model.ngram_size = ngram_size
    model.vocabulary.add("**START**")
    model.vocabulary.add("**END**")
    if(ngram_size == 1):
        model.ngram_counts = defaultdict(int)
    for sent_tokens in sentences:
        if(ngram_size == 1):
            model.ngram_counts["**START**"]+=1
            model.ngram_counts["**END**"]+=1
            for tok in sent_tokens:
                model.ngram_counts[tok] += 1
        else:
            ngrams = make_ngrams(sent_tokens, ngram_size)
            for ngram in ngrams:
                #prefix = tuple(ngram[:ngram_size-1])
                prefix = ngram[0]
                model.ngram_counts[prefix][ngram[-1]] += 1
        for tok in sent_tokens:
            model.vocabulary.add(tok)
    return model


# http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.punkt
sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')

PATH_TO_DATA="/Users/ken77921/Dropbox/course/NLP_TA/for hw1,hw2 - old wordstats, lm assignment/imdb_pos_sample"
fname2content=load_files(PATH_TO_DATA)
movie_review_sents=collect_all_sentences(fname2content,sentence_splitter)

print "Total number of sentences we get: ", len(movie_review_sents)
print movie_review_sents[:5]

uni_gram = get_ngram_counts(movie_review_sents,1)
bi_gram = get_ngram_counts(movie_review_sents,2)


with open('unigram_count_IMDB.json', 'w') as fp:
    json.dump(uni_gram.ngram_counts, fp)

with open('bigram_count_IMDB.json', 'w') as fp:
    json.dump(bi_gram.ngram_counts, fp)
