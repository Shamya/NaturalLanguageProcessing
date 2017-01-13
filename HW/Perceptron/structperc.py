from __future__ import division
import sys,re,random
from collections import defaultdict
from pprint import pprint
from vit import viterbi

##########################
# Stuff you will use

import vit  # your vit.py from part 1
OUTPUT_VOCAB = set(""" ! # $ & , @ A D E G L M N O P R S T U V X Y Z ^ """.split())

##########################
# Utilities

def dict_subtract(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] -= vec2[k]
    return dict(out)

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k]"""
    return max(dct.iterkeys(), key=lambda k: dct[k])

def dict_dotprod(d1, d2):
    """Return the dot product (aka inner product) of two vectors, where each is
    represented as a dictionary of {index: weight} pairs, where indexes are any
    keys, potentially strings.  If a key does not exist in a dictionary, its
    value is assumed to be zero."""
    smaller = d1 if len(d1)<len(d2) else d2  # BUGFIXED 20151012
    total = 0
    for key in smaller.iterkeys():
        total += d1.get(key,0) * d2.get(key,0)
    return total

def read_tagging_file(filename):
    """Returns list of sentences from a two-column formatted file.
    Each returned sentence is the pair (tokens, tags) where each of those is a
    list of strings.
    """
    sentences = open(filename).read().strip().split("\n\n")
    ret = []
    for sent in sentences:
        lines = sent.split("\n")
        pairs = [L.split("\t") for L in lines]
        tokens = [tok for tok,tag in pairs]
        tags = [tag for tok,tag in pairs]
        ret.append( (tokens,tags) )
    return ret
###############################

## Evaluation utilties you don't have to change

def do_evaluation(examples, weights):
    num_correct,num_total=0,0
    for tokens,goldlabels in examples:
        N = len(tokens); assert N==len(goldlabels)
        predlabels = predict_seq(tokens, weights)
        num_correct += sum(predlabels[t]==goldlabels[t] for t in range(N))
        num_total += N
    print "%d/%d = %.4f accuracy" % (num_correct, num_total, num_correct/num_total)
    return num_correct/num_total

def fancy_eval(examples, weights):
    confusion = defaultdict(float)
    bygold = defaultdict(lambda:{'total':0,'correct':0})
    for tokens,goldlabels in examples:
        predlabels = predict_seq(tokens, weights)
        for pred,gold in zip(predlabels, goldlabels):
            confusion[gold,pred] += 1
            bygold[gold]['correct'] += int(pred==gold)
            bygold[gold]['total'] += 1
    goldaccs = {g: bygold[g]['correct']/bygold[g]['total'] for g in bygold}
    for gold in sorted(goldaccs, key=lambda g: -goldaccs[g]):
        print "gold %s acc %.4f (%d/%d)" % (gold,
                goldaccs[gold],
                bygold[gold]['correct'],bygold[gold]['total'],)

def show_predictions(tokens, goldlabels, predlabels):
    print "%-20s %-4s %-4s" % ("word", "gold", "pred")
    print "%-20s %-4s %-4s" % ("----", "----", "----")
    for w, goldy, predy in zip(tokens, goldlabels, predlabels):
        out = "%-20s %-4s %-4s" % (w,goldy,predy)
        if goldy!=predy:
            out += "  *** Error"
        print out

###############################

## YOUR CODE BELOW


def train(examples, stepsize=1, numpasses=10, do_averaging=False, devdata=None):
    """
    IMPLEMENT ME !
    Train a perceptron. This is similar to the classifier perceptron training code
    but for the structured perceptron. Examples are now pairs of token and label
    sequences. The rest of the function arguments are the same as the arguments to
    the training algorithm for classifier perceptron.
    """

    weights = defaultdict(float)
    S = defaultdict(float)

    def get_averaged_weights():
        # IMPLEMENT ME!
        S1={}
        for st in S:
            S1[st] = S[st] * (1/count)
        return dict_subtract(weights, S1)

    count = 0
    for pass_iteration in range(numpasses):
        print "Training iteration %d" % pass_iteration
        # IMPLEMENT THE INNER LOOP!
        # Like the classifier perceptron, you may have to implement code
        # outside of this loop as well!

        for tokens,goldlabels in examples:
            count += 1
            predlabels = predict_seq(tokens, weights)
            goldfeats = features_for_seq(tokens, goldlabels)
            predfeats = features_for_seq(tokens, predlabels)
            
            if predlabels != goldlabels:
                '''
                goldfeats = {goldfeats[k]*(-stepsize) for k in goldfeats}
                predfeats = {predfeats[k]*(stepsize) for k in predfeats}
                weights = dict_subtract(weights, goldfeats)
                weights = dict_subtract(weights, predfeats)
                '''
                g = dict_subtract(goldfeats, predfeats)
                g1={}
                for f in g:
                    g1[f] = g[f] * -stepsize
                weights = dict_subtract(weights, g1)
                
                for f in g:
                    g1[f] = g[f] * (count-1) * -stepsize
                S = dict_subtract(S, g1)

        # Evaluation at the end of a training iter
        print "TR  RAW EVAL:",
        do_evaluation(examples, weights)
        if devdata:
            print "DEV RAW EVAL:",
            do_evaluation(devdata, weights)
        if devdata and do_averaging:
            print "DEV AVG EVAL:",
            do_evaluation(devdata, get_averaged_weights())

    print "Learned weights for %d features from %d examples" % (len(weights), len(examples))

    # NOTE different return value then classperc.py version.
    return weights if not do_averaging else get_averaged_weights()

def predict_seq(tokens, weights):
    """
    IMPLEMENT ME!
    takes tokens and weights, calls viterbi and returns the most likely
    sequence of tags
    """
    #call calc_factor_scores 
    Ascores, Bscores = calc_factor_scores(tokens, weights)
    predlabels = viterbi(Ascores, Bscores, OUTPUT_VOCAB)
    # once you have Ascores and Bscores, could decode with
    # predlabels = greedy_decode(Ascores, Bscores, OUTPUT_VOCAB)
    return predlabels

def greedy_decode(Ascores, Bscores, OUTPUT_VOCAB):
    """Left-to-right greedy decoding.  Uses transition feature for prevtag to curtag."""
    N=len(Bscores)
    if N==0: return []
    out = [None]*N
    out[0] = dict_argmax(Bscores[0])
    for t in range(1,N):
        tagscores = {tag: Bscores[t][tag] + Ascores[out[t-1], tag] for tag in OUTPUT_VOCAB}
        besttag = dict_argmax(tagscores)
        out[t] = besttag
    return out

def local_emission_features(t, tag, tokens):
    """
    Feature vector for the B_t(y) function
    t: an integer, index for a particular position
    tag: a hypothesized tag to go at this position
    tokens: the list of strings of all the word tokens in the sentence.
    Returns a set of features.
    """
    curword = tokens[t]
    feats = {}
    feats["tag=%s_biasterm" % tag] = 1
    feats["tag=%s_curword=%s" % (tag, curword)] = 1
    if curword[0] == '#':
        feats["tag=%s_first=#" % tag] = 1
    if curword[0] == '@':
        feats["tag=%s_first=@" % tag] = 1
    if len(curword) > 1:
        if curword[0] == 'R' and curword[1] == 'T':
            feats["tag=%s_first=RT" % tag] = 1
    if re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', curword):
        feats["tag=%s_URL" % tag] = 1

    return feats

def features_for_seq(tokens, labelseq):
    """
    IMPLEMENT ME!

    tokens: a list of tokens
    labelseq: a list of output labels
    The full f(x,y) function. Returns one big feature vector. This is similar
    to features_for_label in the classifier peceptron except here we aren't
    dealing with classification; instead, we are dealing with an entire
    sequence of output tags.

    This returns a feature vector represented as a dictionary.
    """
    feat = {}
    #emission probabilities
    for i,token in enumerate(tokens):
        fb = local_emission_features(0, labelseq[i], [token])
        for f in fb:
            if f in feat:
                feat[f] += fb[f]
            else:
                feat[f] = fb[f]
    
    #transition probabilities
    last_label = None
    for label in labelseq:
        if last_label is not None:
            trans = "lasttag=%s_curtag=%s" % (last_label, label)
            if trans in feat:
                feat[trans] += 1
            else:
                feat[trans] = 1
                
            
        last_label = label
        
    return feat
    
def calc_factor_scores(tokens, weights):
    """
    IMPLEMENT ME!

    tokens: a list of tokens
    weights: perceptron weights (dict)

    returns a pair of two things:
    Ascores which is a dictionary that maps tag pairs to weights
    Bscores which is a list of dictionaries of tagscores per token
    """
    N = len(tokens)
    # MODIFY THE FOLLOWING LINE
    Ascores = { (tag1,tag2): weights["lasttag=%s_curtag=%s" % (tag1, tag2)] for tag1 in OUTPUT_VOCAB for tag2 in OUTPUT_VOCAB }
    
    Bscores = []
    for t in range(N):
        # IMPLEMENT THE INNER LOOP
        dt= {}
        for tag in OUTPUT_VOCAB:
            feats = local_emission_features(t, tag, tokens)
            f = "tag=%s_curword=%s" % (tag, tokens[t]) 
            fbias =   "tag=%s_biasterm" % tag  
            fhash = "tag=%s_first=#" % tag
            fat = "tag=%s_first=@" % tag
            fRT = "tag=%s_first=RT" % tag
            fURL = "tag=%s_URL" % tag                           
            
            dt[tag] = (feats[f]* weights.get(f,0))+ (feats.get(fhash,0)* weights.get(fhash,0)) + (feats.get(fat,0)* weights.get(fat,0)) + (feats.get(fRT,0)* weights.get(fRT,0)) +(feats.get(fURL,0)* weights.get(fURL,0)) + (feats[fbias] *weights[fbias])

            
        Bscores.append(dt)
    assert len(Bscores) == N
    return Ascores, Bscores

if __name__ == '__main__':
    print local_emission_features(1,'V', ['I','love','cats'])
    print local_emission_features(0,'A', ['Happy','president','Abe'])
    print features_for_seq(['Happy','president','Abe', 'Lincoln'], ['A','N','N','N'])
    
    ''' 
    Run this for output of Q3.5   
    OUTPUT_VOCAB = set("""V P""".split())
    weights = defaultdict(float)
    weights['tag=V_curword=love'] = 1.6
    weights['tag=P_curword=I'] = 0.9
    weights['lasttag=V_curtag=P'] = 1.2
    weights['lasttag=P_curtag=V'] = 0.2
    weights['tag=V_biasterm'] = 0.8
    weights['tag=P_biasterm'] = 1.4
    A, B = calc_factor_scores(['I','love'], weights)
    print A
    print B
    '''
    print predict_seq(['I','love'], defaultdict(float))
    tweet_lst= read_tagging_file('oct27.train.txt')
    print tweet_lst[0][0]
    devdata = read_tagging_file('oct27.dev.txt')
    weights = train(tweet_lst, stepsize=1, numpasses=10, do_averaging=True, devdata=devdata)
    fancy_eval(devdata, weights)