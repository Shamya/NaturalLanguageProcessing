from __future__ import division

import math
import os

from collections import defaultdict


# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

# Path to dataset
PATH_TO_DATA = "/Users/shamya/Documents/Eclipse/NLP/large_movie_review_dataset"
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")


def tokenize_doc(doc):
    """

    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return bow

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }


    def train_model(self, num_docs=None):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.

        num_docs: set this to e.g. 10 to train on only 10 docs from each category.
        """

        if num_docs is not None:
            print "Limiting to only %s docs per clas" % num_docs

        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        print "Starting training with paths %s and %s" % (pos_path, neg_path)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print "REPORTING CORPUS STATISTICS"
        print "NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL]
        print "NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL]
        print "NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL]
        print "NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL]
        print "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab)

    def update_model(self, bow, label):
        """
        IMPLEMENT ME!

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """
        
        words = bow.keys()
        [self.vocab.add(w) for w in words]   
        
        self.class_total_doc_counts[label] += 1 

        for words in bow:
            self.class_total_word_counts[label] += bow[words]
            self.class_word_counts[label][words] += bow[words]

    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """

        Returns the most frequent n tokens for documents with class 'label'.
        """
        return sorted(self.class_word_counts[label].items(), key=lambda (w,c): -c)[:n]

    def p_word_given_label(self, word, label):
        """
        Implement me!

        Returns the probability of word given label (i.e., P(word|label))
        according to this NB model.
        """
        return self.class_word_counts[label][word] / self.class_total_word_counts[label]
        

    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        """
        Implement me!

        Returns the probability of word given label wrt psuedo counts.
        alpha - psuedocount parameter
        """
        return (self.class_word_counts[label][word] + alpha) / (self.class_total_word_counts[label] + (alpha * len(self.vocab)))

    def log_likelihood(self, bow, label, alpha):
        """
        Computes the log likelihood of a set of words give a label and psuedocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; psuedocount parameter
        """
        log_like = 0
        for word,count in bow.items():
            log_like += count*math.log(self.p_word_given_label_and_psuedocount(word, label, alpha))
        
        return log_like
        
    def log_prior(self, label):
        """
        Implement me!

        Returns a float representing the fraction of training documents
        that are of class 'label'.
        """
#         return math.log(self.class_total_doc_counts[label]/sum(self.class_total_doc_counts.values()))    
        return math.log(self.class_total_doc_counts[label]) - math.log(sum(self.class_total_doc_counts.values()))    
        
    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Implement me!

        alpha - psuedocount parameter
        bow - a bag of words (i.e., a tokenized document)
        Computes the unnormalized log posterior (of doc being of class 'label').
        """
        return self.log_prior(label) + self.log_likelihood(bow, label, alpha)

    def classify(self, bow, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        bow - a bag of words (i.e., a tokenized document)

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior).
        """
        pos = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        neg = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)
        
        return NEG_LABEL if pos < neg else POS_LABEL

    def likelihood_ratio(self, word, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        ps = self.p_word_given_label_and_psuedocount(word, POS_LABEL, alpha)
        ng = self.p_word_given_label_and_psuedocount(word, NEG_LABEL, alpha)
        return ps/ng

    def evaluate_classifier_accuracy(self, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        pos_path = os.path.join(TEST_DIR, POS_LABEL)
        neg_path = os.path.join(TEST_DIR, NEG_LABEL)
        count = 0
        correct = 0
        scores = {'tp':0,'tn':0,'fp':0,'fn':0}
        print "Starting testing with alpha=%.1f" % alpha
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            for f in filenames:
                count += 1
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    bow = tokenize_doc(content)
                pred = nb.classify(bow, alpha)
                
                if pred == label:
                    correct += 1 
                    
                #precision and recall
                #true positive 
                if label == POS_LABEL:
                    if pred == POS_LABEL:
                        scores['tp'] += 1
                    else:
                        scores['fn'] += 1
                else:
                    if pred == POS_LABEL:
                        scores['fp'] += 1
                    else:
                        scores['tn'] += 1
                        
        test_acc = correct / count 
        precision = scores['tp'] / (scores['tp'] + scores['fp'])
        recall = scores['tp'] / (scores['tp'] + scores['fn'])
        
        z_score = (2*precision*recall)/(precision+recall)
        return test_acc, z_score

def produce_hw1_results():
    # PRELIMINARIES

    # QUESTION 1.1
    # uncomment the next two lines when ready to answer question 1.2
    print "VOCABULARY SIZE: " + str(len(nb.vocab))
    print ''

    # QUESTION 1.2
    # uncomment the next set of lines when ready to answer qeuestion 1.2
    print "TOP 10 WORDS FOR CLASS " + POS_LABEL + " :"
    for tok, count in nb.top_n(POS_LABEL, 10):
        print '', tok, count
    print ''

    # print "TOP 10 WORDS FOR CLASS " + NEG_LABEL + " :"
    for tok, count in nb.top_n(NEG_LABEL, 10):
        print '', tok, count
    print ''
    print '[done.]'

def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuries. You may want to modify this function
    to enhance your plot.
    """

    import matplotlib.pyplot as plt

    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.show()

def plot_psuedocount_vs_zscore(psuedocounts, z_score):
    """
    A function to plot psuedocounts vs. accuries. You may want to modify this function
    to enhance your plot.
    """

    import matplotlib.pyplot as plt

    plt.plot(psuedocounts, z_score)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Z score')
    plt.title('Psuedocount Parameter vs. Z score')
    plt.show()

if __name__ == '__main__':
    nb = NaiveBayes()
    nb.train_model() #1.1
    
    print nb.top_n(POS_LABEL, 10)   #1.2
    print nb.top_n(NEG_LABEL, 10)   #1.2
    
    print "(fantastic, +)" , nb.p_word_given_label("fantastic", POS_LABEL) #2.1
    print "(fantastic, -)" , nb.p_word_given_label("fantastic", NEG_LABEL)
    print "(boring, +)" , nb.p_word_given_label("boring", POS_LABEL)
    print "(boring, -)" , nb.p_word_given_label("boring", NEG_LABEL)
    
    accuracies = []
    z_score = []
    best_acc = 0
    best_alpha = None
    psuedocounts = [0.3,0.5,1,3,5,8,10,12,14,15,17,20,22,25]
    for alpha in psuedocounts:
        acc, z = nb.evaluate_classifier_accuracy(alpha)
        if acc > best_acc:
            best_alpha = alpha
        accuracies.append(acc)
        z_score.append(z)
        
    print accuracies
    print z_score
    
    plot_psuedocount_vs_accuracy(psuedocounts, accuracies)
    plot_psuedocount_vs_zscore(psuedocounts, z_score)
    
    print "fantastic LR: " , nb.likelihood_ratio('fantastic', 1.0)
    print "boring LR: " , nb.likelihood_ratio('boring', 1.0)
    print "the LR: " , nb.likelihood_ratio('the', 1.0)
    print "to LR: " , nb.likelihood_ratio('to', 1.0)
    
    produce_hw1_results()