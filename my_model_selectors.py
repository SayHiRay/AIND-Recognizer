import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        log_N = np.log(len(self.lengths))
        min_bic = float('inf')
        best_model = None

        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_hidden_states)
                log_L = model.score(self.X, self.lengths)
                p = model.startprob_.size + model.transmat_.size\
                    + model.means_.size + model.covars_.diagonal().size
                bic = -2 * log_L + p * log_N
                if bic < min_bic:
                    best_model = model
                    min_bic = bic
            except:
                pass
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_dic = float("-inf")
        best_model = None
        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_hidden_states)
                log_L = model.score(self.X, self.lengths)

                partial_sum = 0
                i = 0
                for word in self.words:
                    if word != self.this_word:
                        X, lengths = self.hwords[word]
                        try:
                            partial_sum += model.score(X, lengths)
                            i += 1
                        except:
                            pass
                if i > 0:
                    partial_mean = partial_sum / i
                else:
                    partial_mean = 0

                bic = log_L - partial_mean
                best_dic, best_model = max((best_dic, best_model),
                                           (bic, model))
            except:
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        # warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        max_mean_log_L = float('-inf')
        best_num_hidden_states = None
        n_splits = min(3, len(self.lengths))

        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            sum_log_L = 0
            if n_splits < 2:
                try:
                    self.X, self.lengths = self.hwords[self.this_word]
                    model = self.base_model(num_hidden_states)
                    max_mean_log_L = model.score(self.X, self.lengths)
                except:
                    pass
            else:
                split_method = KFold(n_splits=n_splits)
                i = 0
                for cv_train_idx, cv_test_idx in split_method.split(
                        self.sequences):
                    try:
                        self.X, self.lengths = combine_sequences(cv_train_idx,
                                                                 self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx,
                                                                 self.sequences)

                        model = self.base_model(num_hidden_states)

                        sum_log_L = + model.score(self.X, self.lengths)
                        i += 1
                    except:
                        pass
                if i > 0:
                    mean_log_L = sum_log_L / i
                else:
                    mean_log_L = 0
                if mean_log_L > max_mean_log_L:
                    max_mean_log_L = mean_log_L
                    best_num_hidden_states = num_hidden_states

        self.X, self.lengths = self.hwords[self.this_word]
        if best_num_hidden_states is None:
            return self.base_model(best_num_hidden_states)
        else:
            return self.base_model(self.n_constant)


