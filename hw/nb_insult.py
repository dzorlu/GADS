#!/usr/local/bin/python

import nltk
import pandas as pd
import sys

class InsultDetector:
    """class to identify comments as insults"""
    def __init__(self, comment_arr, insult_arr):
        """constructor takes two input arrays:
        comment_arr: an array of strings, comments in a conversation
        insult_arr:  each entry is True iff corresponding comment is an insult
        """

        # constructor initializes a few key instance variables
        #
        # prior_mean: the prior for a comment to be an insult
        # prior_nice: the prior for a comment to NOT be an insult
        # all_words:  a set of all words we've seen
        # mean_probs: a dictionary of words -> doubles
        #             word -> prob(word \in comment | comment is insult)
        # nice_probs: word -> prob(word \in comment | comment is NOT an insult)
        # 
        # NB: our word probabilities represent whether a word appears at all
        #     in a given comment.  They don't keep track of multiplicity.
        #     Also, we don't create any sort of model to accommodate words
        #     that were not encountered in the training set.  

        assert len(comment_arr) == len(insult_arr)

        num_all  = len(insult_arr)
        num_mean = sum(insult_arr)
        num_nice = num_all - num_mean

        self.prior_mean = num_mean * 1.0 / num_all
        self.prior_nice = num_nice * 1.0 / num_all

        # first maintain a count of how many mean/nice comments
        # contain a given word (not tracking multiplicity)
        mean_words = {}
        nice_words = {}

        self.all_words = set()

        for idx, comment in enumerate(comment_arr):

            tokens = nltk.tokenize.word_tokenize(comment)
            # we only consider nice alphabetical tokens.  This might
            # wreak havoc on heavily escaped text, but in that case we
            # should probably make use of a more sophisticated tokenizer
            words = set([w.lower() for w in tokens if w.isalpha()])
    
            for word in words:
                self.all_words.add(word)
                if insult_arr[idx]:
                    mean_words[word] = mean_words.get(word, 0) + 1
                else:
                    nice_words[word] = nice_words.get(word, 0) + 1

        # from the counts, compute the probabilities
        self.mean_probs = {}
        self.nice_probs = {}

        for word, count in mean_words.iteritems():
            self.mean_probs[word] = count*1.0/num_mean
        for word, count in nice_words.iteritems():
            self.nice_probs[word] = count*1.0/num_nice

    def is_insult(self, comment):
        """returns true if the comment is perceived to be an insult."""

        tokens = nltk.tokenize.word_tokenize(comment)
        words = set([w.lower() for w in tokens if w.lower() in self.all_words])
        # note that we ignore unfamiliar words - they have probability 0

        # the relative probabilities that the comment is an insult or not
        rel_mean = self.prior_mean
        rel_nice = self.prior_nice

        for word in words:
            rel_mean *= self.mean_probs.get(word, 0)
            rel_nice *= self.nice_probs.get(word, 0)

        # is it more likely an insult or not?
        return rel_mean > rel_nice

if __name__ == "__main__":
    """Takes two arguments to run: 
    1. a filename for the training set
    2. a filename for the test set
    """

    train_data = pd.read_csv(sys.argv[-2])
    test_data  = pd.read_csv(sys.argv[-1])

    id = InsultDetector(train_data.Comment, train_data.Insult)

    # keep track of results for easy access if %run from iPython
    results = [None]*len(test_data)

    # print out the input file, prepending a column for Insult prediction
    print "Insult," + ",".join(test_data.columns)
    for ix, row in test_data.iterrows():
        tail = ",".join([str(x) for x in row.values])
        results[ix] = int(id.is_insult(row['Comment']));
        print str(results[ix]) + "," + tail
        
