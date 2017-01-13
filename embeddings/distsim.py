from __future__ import division
import sys,json,math
import os
import numpy as np
from operator import itemgetter

def load_word2vec(filename):
    # Returns a dict containing a {word: numpy array for a dense word vector} mapping.
    # It loads everything into memory.
    
    w2vec={}
    with open(filename,"r") as f_in:
        for line in f_in:
            line_split=line.replace("\n","").split()
            w=line_split[0]
            vec=np.array([float(x) for x in line_split[1:]])
            w2vec[w]=vec
    return w2vec

def load_contexts(filename):
    # Returns a dict containing a {word: contextcount} mapping.
    # It loads everything into memory.

    data = {}
    for word,ccdict in stream_contexts(filename):
        data[word] = ccdict
    print "file %s has contexts for %s words" % (filename, len(data))
    return data

def stream_contexts(filename):
    # Streams through (word, countextcount) pairs.
    # Does NOT load everything at once.
    # This is a Python generator, not a normal function.
    for line in open(filename):
        word, n, ccdict = line.split("\t")
        n = int(n)
        ccdict = json.loads(ccdict)
        yield word, ccdict

def cossim_sparse(v1,v2):
    # Take two context-count dictionaries as input
    # and return the cosine similarity between the two vectors.
    # Should return a number beween 0 and 1

    num = 0
    din1 = 0
    din2 = 0
    for k in v1:
        num += v1[k] * v2.get(k,0)
        din1 += v1[k]**2
    for k in v2:
        din2 += v2[k]**2

    return num / (np.sqrt(din1)*np.sqrt(din2))
    

def cossim_dense(v1,v2):
    # v1 and v2 are numpy arrays
    # Compute the cosine simlarity between them.
    # Should return a number between -1 and 1
    
    ## TODO: delete this line and implement me
    return np.sum(v1*v2)/(np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))

def show_nearest(word_2_vec, w_vec, exclude_w, sim_metric):
    #word_2_vec: a dictionary of word-context vectors. The vector could be a sparse (dictionary) or dense (numpy array).
    #w_vec: the context vector of a particular query word `w`. It could be a sparse vector (dictionary) or dense vector (numpy array).
    #exclude_w: the words you want to exclude in the responses. It is a set in python.
    #sim_metric: the similarity metric you want to use. It is a python function
    # which takes two word vectors as arguments.
    similar = {}
    count = 0
    for w_check in word_2_vec:
        if w_check not in exclude_w:
            d = sim_metric(word_2_vec[w_check], w_vec)
            if count < 21:
                similar[w_check] = d
                count += 1
            else:
                min_w = min(similar, key = similar.get)
                if d > similar[min_w]:
                    similar[w_check] = d
                    del similar[min_w]
    
    for k, v in sorted(similar.items(), key=itemgetter(1), reverse=True):
        print k, ":", v
