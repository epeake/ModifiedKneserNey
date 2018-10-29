#!/usr/bin/env python3
"""
Module contains the contents necessary to perform a modified Kneser-Ney smoothing and score the average ngram log
probabilities of additional corpuses.

"""
from re import sub as re_sub
from nltk import pos_tag, ngrams
from nltk.corpus import wordnet
from nltk.data import load as nltk_load
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from numpy import multiply as np_multiply
from numpy import sum as np_sum
from math import log

__author__ = "Elijah Peake"
__email__ = "elijah.peake@gmail.com"


class ModifiedKneserNey:
    """
    A modified, interpolated Kneser-Ney smoothing object with corrections out-of-vocabulary words and small data sets.
    Words are automatically lemmatized unless specified otherwise in train().

    """

    def __init__(self):
        """
        Empty constructor for KneserNey.

        """
        self.lemmatized = None  # sees if user wants to lemmatize
        self.corpus = None
        self.highest_order = None
        self.ngram_probabilities = None
        self.av_unk_probability = None
        self.vocab = None

    def _get_wordnet_pos(self, treebank_tag):
        """
        Taken from: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python

        Needed for _lemmatize function.

        :param treebank_tag: from nltk.pos_tag(tokens)
        :return: wordnet part of speach
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ

        elif treebank_tag.startswith('V'):
            return wordnet.VERB

        elif treebank_tag.startswith('N'):
            return wordnet.NOUN

        elif treebank_tag.startswith('R'):
            return wordnet.ADV

        else:
            return ""

    def _lemmatize(self, tokens):
        """
        Lemmatizes tokens

        :param tokens
        :return: lemmatized tokens
        """
        treebank_tag = pos_tag(tokens)
        word_tags = [self._get_wordnet_pos(treebank_tag[i][1])
                     for i in range(len(treebank_tag))]
        lemmatized_words = []
        lmtzr = WordNetLemmatizer()
        for i in range(0, len(treebank_tag)):
            if word_tags[i] != "":  # if it has a tag useful to the lemmatizer
                lemmatized_words.append(lmtzr.lemmatize(
                    treebank_tag[i][0], word_tags[i]))

            else:
                lemmatized_words.append(lmtzr.lemmatize(treebank_tag[i][0]))

        return lemmatized_words

    def _get_padded_ngrams(self, corpus, n):
        """
        Takes a corpus and replaces all punctuation that does not change word meaning and standardizes all sentence
        ending punctuation with start and stop markers and returns list of padded ngrams.

        start marker = <s>
        stop marker =  <\s>

        :param corpus:  String.  Corpus to be padded
        :return: list of padded ngrams
        """
        # remove non-word joining parentheses (back)
        edited_corpus = re_sub("(?<![a-zA-Z])-", " ", corpus)

        # remove non-word joining parentheses (forward)
        edited_corpus = re_sub("-(?![a-zA-Z])", " ", edited_corpus)

        # removing all non-information punctuation
        edited_corpus = re_sub("[^?.\-'!\w]|\(|\)", " ", edited_corpus)

        # remove numbers
        edited_corpus = re_sub("[0-9]*", "", edited_corpus)

        # replace multiple sentence finishers with single ones
        edited_corpus = re_sub("([.?!]+\s*)+", ". ", edited_corpus)

        tokenizer = nltk_load('tokenizers/punkt/english.pickle')
        sentence_corpus = tokenizer.sentences_from_text(edited_corpus)
        n_grams = []
        for sentence in sentence_corpus:
            sentence = sentence.rstrip('.')
            tokens = sentence.lower().split()
            if self.lemmatized:
                tokens = self._lemmatize(tokens)
            tmp_ngrams = ngrams(tokens, n, pad_left=True, pad_right=True,
                                left_pad_symbol='<s>', right_pad_symbol="</s>")
            n_grams.extend(tmp_ngrams)

        return n_grams

    def _get_all_padded_ngrams(self):
        """
        Finds padded ngrams for each for degrees 1 to highest order.

        :return: padded ngrams
        """
        return [self._get_padded_ngrams(self.corpus, i)
                for i in range(1, self.highest_order + 1)]

    def _calc_ngram_freqs(self, padded_ngrams):
        """
        Finds the frequencies of each ngram for degrees 1 to highest order.

        :return: ngram frequencies
        """
        ngram_freqs = []
        for i in range(0, self.highest_order):
            if i != 0:
                freqs = Counter(padded_ngrams[i])

                # making our set of unknown values
                unique_degree_lower = set()
                for n_grams in freqs.keys():
                    unique_degree_lower.add(n_grams[:-1])

                to_add = [(*j, "<unk>")
                          for j in unique_degree_lower]

                # dictionary of all zeros as counts
                unknown_dict = dict(zip(to_add, [0] * len(to_add)))
                unknown_dict.update(freqs)

                # sorting our dictionary
                sorted(unknown_dict.items(), key=lambda x: x[1])
                ngram_freqs.append(unknown_dict)

            else:
                freqs = Counter(padded_ngrams[i])
                unknown_dict = {("<unk>",): 0}
                unknown_dict.update(freqs)

                # sorting our dictionary
                sorted(unknown_dict.items(), key=lambda x: x[1])
                ngram_freqs.append(unknown_dict)

        # need to account for the end pad (probability that a sentence will end)
        if self.highest_order > 1:
            # fun with generators
            end_of_sentence_count = sum(ngram_freqs[1].get(key) for key in ngram_freqs[1]
                                        if key[-1] == "</s>")
            ngram_freqs[0].update(({("</s>",): end_of_sentence_count}))

        return ngram_freqs

    def _calc_discounts(self, ngram_freqs):
        """
        Estimates our discount amount based on our training data.  Estimation proposed as our "usual" method of
        estimation in "On the Estimation of Discount Parameters for Language Model Smoothing" by Sundermeyer et al.

        :return: tupple of discount amounts for 1grams, 2grams, and 3+grams
        """
        try:
            n = self.highest_order
            freq_1 = ngram_freqs[0]
            n1 = 0
            n2 = 0
            for value in freq_1.values():
                if value == 1:
                    n1 += 1

                if value == 2:
                    n2 += 1

            one_gram_discount = n1 / (n1 + (2 * n2))
            if n >= 2:
                freq_2 = ngram_freqs[1]
                n1 = 0
                n2 = 0
                n3 = 0
                for value in freq_2.values():
                    if value == 1:
                        n1 += 1

                    if value == 2:
                        n2 += 1

                    if value == 3:
                        n3 += 1

                b = n1 / (n1 + (2 * n2))
                two_gram_discount = 2 - (3 * b * (n3 / n2))

            if n == 1:  # 1 is our highest order
                return one_gram_discount, 0, 0

            elif n == 2:
                return one_gram_discount, two_gram_discount, 0

            elif n >= 3:  # in this case we need our 3+ discount
                freq_3 = ngram_freqs[2]
                n1 = 0
                n2 = 0
                n3 = 0
                n4 = 0
                for value in freq_3.values():
                    if value == 1:
                        n1 += 1

                    if value == 2:
                        n2 += 1

                    if value == 3:
                        n3 += 1

                    if value == 4:
                        n4 += 1

                b = n1 / (n1 + (2 * n2))
                three_gram_discount = 3 - (4 * b * (n4 / n3))
                return one_gram_discount, two_gram_discount, three_gram_discount

        except ZeroDivisionError:
            print("More training data required and or a lower highest order")
            raise

    def _get_vocab(self, padded_ngrams):
        """
        Finds our total working vocab, including our start and stop pads.

        :return: our working vocab
        """
        vocab = set(padded_ngrams[0])  # 1 grams
        vocab.add(('</s>',))
        vocab.add(('<s>',))
        return vocab

    def _calc_ngram_types(self, n, padded_ngrams):
        """
        Calculates the number of distinct n grams for order n

        :return: number of distinct ngrams
        """
        return len(set(padded_ngrams[n - 1]))  # -1 due to indexing

    def _get_ngram_types(self, padded_ngrams):
        """
        Gets the number of of distinct n grams for orders 1 to highest order.

        :return: list of distinct ngrams for each order
        """
        return [self._calc_ngram_types(i, padded_ngrams)
                for i in range(1, self.highest_order + 1)]

    def _calc_unique_and_count(self, n, ngram_freqs):
        """
        Calculates the number of contexts for each n-1gram that appear exactly once,
        thus n must be greater than 1.  Additionally calculates total frequencies of all n-1grams.


        :return: List of two dictionaries, one for unique and one for count
        """
        unique_key = []
        unique = {}

        # get number of unique contexts
        keys = [key[:-1]
                for key in ngram_freqs[n - 1].keys()]
        key_number = 1
        key_index = 0
        previous_key = keys[key_index]
        for key in keys:
            if key == previous_key:
                unique_key.append(key_number)
            else:
                previous_key = keys[key_index]
                key_number += 1
                unique_key.append(key_number)
            key_index += 1

        # going to count the number of occurrences of each unique gram
        for i in range(1, max(unique_key) + 1):
            index = unique_key.index(i)
            key = keys[index]
            # we subtract one because of our <unk>
            value = (unique_key.count(i) - 1)
            unique.update({key: value})

        # now onto counts...
        values = [val for val in ngram_freqs[n - 1].values()]
        count = {key: value for (key, value) in unique.items()}
        sums = []
        key_number = 1
        val_sum = 0
        for i in range(0, len(values)):
            if key_number == unique_key[i]:
                val_sum += values[i]

            else:
                sums.append(val_sum)
                val_sum = values[i]
                key_number += 1

        # because the last val_sum will not be added because doesn't meet else condition
        sums.append(val_sum)
        index = 0
        for key in count:
            count[key] = sums[index]
            index += 1

        return unique, count

    def _get_unique_and_count(self, ngram_freqs):
        """
        Gets unique n-1grams and n-1gram count for n-grams orders 2 to highest order

        :return: List of highest order - 1 lists with two elements, one for unique and one for count
        """
        if self.highest_order > 1:  # because n-1grams
            return [self._calc_unique_and_count(i, ngram_freqs)
                    for i in range(2, self.highest_order + 1)]

        else:
            return None

    def _calc_adj_probs(self, n, ngram_freqs, discounts, ngram_types, unique_and_count):
        """
        Calculates the discounted probabilities of each ngram

        :param n: int.  Order of ngram.
        :return: discounted probabilities
        """
        if n == 1:
            new_values = []
            val_sum = sum(val for val in ngram_freqs[0].values())
            discount = discounts[0]
            contexts = ngram_types[0]
            for key in ngram_freqs[0]:
                val = ngram_freqs[0].get(key)
                first_term = max(val - discount, 0) / val_sum
                second_term = discount / contexts
                new_values.append(first_term + second_term)

            return new_values

        else:
            new_values = []
            first_terms = []
            second_terms = []

            discount = discounts[n - 1]
            our_sum = unique_and_count[n - 2][1]
            unique = unique_and_count[n - 2][0]
            contexts = ngram_types[n - 1]

            for key in ngram_freqs[n - 1]:
                val = ngram_freqs[n - 1].get(key)
                try:
                    first_term = max(val - discount, 0) / our_sum.get(key[:-1])
                except ZeroDivisionError:
                    print("More training data required and or a lower highest order")
                    raise
                second_term = discount * (unique.get(key[:-1]) / contexts)
                first_terms.append(first_term)
                second_terms.append(second_term)

            new_values.append(first_terms)
            new_values.append(second_terms)

            return new_values

    def _update_freqs(self, ngram_freqs, discounts, ngram_types, unique_and_count):
        """
        Converts our frequencies into probabilities

        :return: probabilities for each ngram (non-smoothed but discounted)
        """
        vals = [self._calc_adj_probs(i, ngram_freqs, discounts, ngram_types, unique_and_count)
                for i in range(1, self.highest_order + 1)]
        for i in range(0, self.highest_order):
            if i == 0:
                index = 0
                for key in ngram_freqs[i]:
                    ngram_freqs[i][key] = vals[0][index]
                    index += 1

            else:
                index = 0
                for key in ngram_freqs[i]:
                    ngram_freqs[i][key] = vals[i][0][index], vals[i][1][index]
                    index += 1

        return ngram_freqs

    def _handle_end_pad(self, ngram_freqs):
        """
        To avoid none types when interpolating

        :return: new ngram_freqs
        """
        if self.highest_order > 2:
            for i in range(2, self.highest_order):
                without_double_end = {}
                for key in ngram_freqs[i]:
                    if (key[-2:] != ('</s>', '</s>') and  # we need to filter these cases out for equal ngram set sizes
                            key[-2:] != ('</s>', '<unk>')):
                        without_double_end.update({key: ngram_freqs[i].get(key)})

                ngram_freqs[i] = without_double_end

    def _interpolate(self, ngram_freqs):
        """
        Finds new, smoothed ngram probabilities

        :return: ngram probabilities of highest order
        """
        keys = [key for key in ngram_freqs[self.highest_order - 1]]
        all_values = []

        # makes it so that the ngram goes down in order
        n_gram_cutoff = self.highest_order - 1
        for i in range(0, self.highest_order):
            first_terms = []
            second_terms = []
            for j in range(0, len(keys)):
                key_of_interest = keys[j][n_gram_cutoff:]
                val = ngram_freqs[i].get(key_of_interest)
                if i == 0:
                    first_terms.append(val)

                else:
                    first_terms.append(val[0])
                    second_terms.append(val[1])

            if i == 0:
                all_values.append(first_terms)

            else:
                all_values.append([first_terms, second_terms])

            n_gram_cutoff -= 1

        score = []
        for i in range(0, self.highest_order):
            if i == 0:
                score = all_values[0]

            else:
                score = np_multiply(score, all_values[i][1]) + all_values[i][0]

        probabilities = {}
        index = 0
        for key in ngram_freqs[self.highest_order - 1]:
            probabilities.update({key: score[index]})
            index += 1

        return probabilities

    def _find_av_unk_probability(self):
        """
        Finds the average probability of an unknown ngram appearing

        :return: average probability
        """
        probability_sum = 0
        count = 0
        for key in self.ngram_probabilities:
            if key[-1] == "<unk>":
                probability_sum += self.ngram_probabilities.get(key)
                count += 1

        return probability_sum / count

    def train(self, corpus, highest_order, lemmatize=True):
        """
        Trains/initializes our Kneser-Ney object.

        :param corpus: String.  An ASCII encoded corpus of data to train the smoothing algorithm.
        :param highest_order: Int. Desired highest order of n-gram
        :param lemmatize: Boolean.  Good for smaller data sets, choose whether or not to lemmatize words for ngrams.
        :return: VOID
        """
        self.lemmatized = lemmatize
        self.corpus = corpus
        self.highest_order = highest_order
        padded_ngrams = self._get_all_padded_ngrams()
        ngram_freqs = self._calc_ngram_freqs(padded_ngrams)
        discounts = self._calc_discounts(ngram_freqs)
        self.vocab = self._get_vocab(padded_ngrams)
        ngram_types = self._get_ngram_types(padded_ngrams)
        unique_and_count = self._get_unique_and_count(ngram_freqs)

        # turning freqs to probabilities
        self._update_freqs(ngram_freqs, discounts, ngram_types, unique_and_count)
        self._handle_end_pad(ngram_freqs)
        self.ngram_probabilities = self._interpolate(ngram_freqs)
        self.av_unk_probability = self._find_av_unk_probability()

    def log_score_per_ngram(self, corpus):
        """
        Given a corpus outside of training data.  Finds the average ngram log probability of the corpus.

        :param corpus: String.  ASCII encoded corpus to score
        :return: average ngram log probability
        """
        probability_keys = self._get_padded_ngrams(corpus, self.highest_order)
        for i in range(0, len(probability_keys)):
            if (probability_keys[i][-1],) not in self.vocab:
                probability_keys[i] = *probability_keys[i][:-1], "<unk>"

        sentence_probabilities = [self.ngram_probabilities.get(key)
                                  for key in probability_keys]
        for i in range(0, len(sentence_probabilities)):

            # this is the case for completely unknown
            if sentence_probabilities[i] is None:
                sentence_probabilities[i] = log(self.av_unk_probability)

            else:
                sentence_probabilities[i] = log(sentence_probabilities[i])

        log_sum = np_sum(sentence_probabilities)
        all_ngrams = []
        all_ngrams.extend(ngrams(corpus.split(), self.highest_order))
        ngram_count = len(all_ngrams)
        if not ngram_count:
            print("Error: Not enough ngrams.  Ensure that corpus contains at least as many words as the highest order")
            return float("-inf")  # this case is impossible

        return log_sum / ngram_count

    def __repr__(self):
        corpus_len = None
        if self.corpus:
            corpus_len = len(self.corpus.split())

        return (f"{self.__class__.__name__}("
                f"lemmatized = {self.lemmatized!r}, "
                f"highest_order = {self.highest_order!r}, "
                f"training_corpus_word_count = {corpus_len!r})")
