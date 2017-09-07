import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from preprocessing.utils_pre import get_hash_indices
from preprocessing.utils_pre import binary_encode as CODE
from scipy import sparse
import numpy as np


english = stopwords.words('english')


class IndexVectorizer:

    def __init__(self, vocab_index=None, sparsity_code_size=16, tokenizer_sep=","):
        self.inv_vocab_index = dict()
        if vocab_index is None:
            self.vocab_index = dict()
            self.current_index = 1  # 0 is reserved for empty
        else:
            self.vocab_index = vocab_index
            for k, v in vocab_index.items():
                self.inv_vocab_index[v] = k
            self.current_index = len(self.vocab_index) + 1
        self.tokenizer = lambda text: tokenize(text, tokenizer_sep)
        self.stop_words = english
        self.code_dim_int = sparsity_code_size
        self.code_dim = self.code_dim_int * 32

    def get_vocab_dictionaries(self):
        return self.vocab_index, self.inv_vocab_index

    def add_new_term(self, w):
        self.vocab_index[w] = self.current_index
        self.inv_vocab_index[self.current_index] = w
        self.current_index += 1

    def transform(self, list_text):
        for text in list_text:  # we expect only one el in list in IndexVectorizer for now
            list_tokens = self.tokenizer(text)
            list_tokens = [x for x in list_tokens if x not in self.stop_words and len(x) > 2]
            code_vector = np.asarray([0] * self.code_dim_int, dtype=np.int32)
            for t in list_tokens:
                if t not in self.vocab_index:  # make sure word is in vocab
                    self.add_new_term(t)
                indices = get_hash_indices(t, self.code_dim_int)
                for idx in indices:
                    if code_vector[idx] == 0:
                        code_vector[idx] = self.vocab_index[t]  #/ float_embedding_factor
                        continue  # on success we stop trying to insert
            bin_code_vector = CODE(code_vector)
            sparse_bin_code_vector = sparse.csr_matrix(bin_code_vector)
            return sparse_bin_code_vector


class FlatIndexVectorizer:

    def __init__(self, vocab_index=None, sparsity_code_size=16):
        self.inv_vocab_index = dict()
        if vocab_index is None:
            self.vocab_index = dict()
            self.current_index = 1  # 0 is reserved for empty
        else:
            self.vocab_index = vocab_index
            for k, v in vocab_index.items():
                self.inv_vocab_index[v] = k
            self.current_index = len(self.vocab_index) + 1
        #self.tokenizer = lambda text: tokenize(text, tokenizer_sep)
        #self.stop_words = english
        self.code_dim_int = sparsity_code_size
        self.code_dim = self.code_dim_int * 32

    def get_vocab_dictionaries(self):
        return self.vocab_index, self.inv_vocab_index

    def add_new_term(self, w):
        self.vocab_index[w] = self.current_index
        self.inv_vocab_index[self.current_index] = w
        self.current_index += 1

    def transform(self, list_text):
        list_text = list_text[0].split(" ")  # split by space
        list_text = [el.strip() for el in list_text]  # clean spaces

        code_vector = np.asarray([0] * self.code_dim_int, dtype=np.int32)

        for text in list_text:  # we expect only one el in list in IndexVectorizer for now
            #list_tokens = self.tokenizer(text)
            #list_tokens = [x for x in list_tokens if x not in self.stop_words and len(x) > 2]


            #for t in list_tokens:
            if text not in self.vocab_index:  # make sure word is in vocab
                self.add_new_term(text)
            indices = get_hash_indices(text, self.code_dim_int)
            for idx in indices:
                if code_vector[idx] == 0:
                    code_vector[idx] = self.vocab_index[text]  #/ float_embedding_factor
                    continue  # on success we stop trying to insert
        bin_code_vector = CODE(code_vector)
        sparse_bin_code_vector = sparse.csr_matrix(bin_code_vector)
        return sparse_bin_code_vector


class CustomVectorizer:

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def get_vector_for_tuple(self, tuple):
        vector = self.vectorizer.transform([tuple])
        return vector  # vector may or may not be sparse

    def get_vocab_dictionaries(self):  # just calling parent
        return self.vectorizer.get_vocab_dictionaries()


def get_sample_from_tokens(tokens, vectorizer):
    return


def tokenize(tuple_str, separator, min_token_length=3):
    clean_tokens = list()
    tokens = tuple_str.split(separator)
    for t in tokens:
        t = t.lower()
        if len(t) < min_token_length:
            continue
        if re.search('[0-9]', t) is not None:
            continue
        t = t.replace('_', ' ')  # testing
        t = t.replace('-', ' ')
        t = t.replace(',', ' ')
        t = t.lower()
        t_tokens = t.split(' ')
        for token in t_tokens:
            if token == '' or len(token) < min_token_length:
                continue
            clean_tokens.append(token)
    return list(clean_tokens)


if __name__ == "__main__":
    print("Text processor")

    mit_dwh_vocab = utils_pre.get_tf_dictionary("/Users/ra-mit/development/fabric/data/statistics/mitdwhall_tf_only")

    # i = 0
    # terms = []
    # for k, v in mit_dwh_vocab.items():
    #     if i < 8:
    #         terms.append(k)
    #         i += 1
    # query = terms.join(' ')

    import six

    indices = set(mit_dwh_vocab.values())
    if len(indices) != len(mit_dwh_vocab):
        raise ValueError("Vocabulary contains repeated indices.")
    for i in range(len(mit_dwh_vocab)):
        if i not in indices:
            print("MISSING INDEX: " + str(i))

    sor = sorted(mit_dwh_vocab.items(), key=lambda x: x[1], reverse=False)

    tf_vectorizer = CountVectorizer(max_df=1., min_df=0,
                                    encoding='latin1',
                                    tokenizer=lambda text: tokenize(text, " "),
                                    vocabulary=mit_dwh_vocab,
                                    stop_words='english')

    x = tf_vectorizer.transform(["whigham guoxing ellsberry nan gballard", "mit srmadden"])

    print(str(type(x)))

    print(str(x))
