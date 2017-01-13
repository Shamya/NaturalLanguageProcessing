import random

def weighted_draw_from_dict(prob_dict):
    # Utility function -- do not modify
    # Randomly choose a key from a dict, where the values are the relative probability weights.
    # http://stackoverflow.com/a/3679747/86684
    choice_items = prob_dict.items()
    total = sum(w for c, w in choice_items)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choice_items:
       if upto + w > r:
          return c
       upto += w
    assert False, "Shouldn't get here"


## ---------------------- write your answers below here -------------------


def draw_next_word_unigram_model(uni_counts):
    c = weighted_draw_from_dict(uni_counts)
    return c 

def draw_next_word_bigram_model(uni_counts, bi_counts, prev_word):
    c = weighted_draw_from_dict(bi_counts[prev_word])
    return c 

def sample_sentence(uni_counts, bi_counts):
    tokens = ['**START**']
    tk = '**START**'

    #add tokens until end
    while tk != '**END**':
        tk = draw_next_word_bigram_model(uni_counts, bi_counts, tk)
        tokens.append(tk)

    return tokens

