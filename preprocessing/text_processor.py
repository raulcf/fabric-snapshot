import re
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import utils_pre
from nltk.corpus import stopwords
from preprocessing.utils_pre import get_hash_indices, binary_encode as CODE
from scipy import sparse
import numpy as np


english = stopwords.words('english')


class IndexVectorizer:

    def __init__(self):
        self.vocab_index = dict()
        self.inv_vocab_index = dict()
        self.current_index = 0
        self.tokenizer = lambda text: tokenize(text, ",")
        self.stop_words = english
        self.code_dim_int = 8  # FIXME: let someone configure this
        self.code_dim = self.code_dim_int * 32

    def get_vocab_dictionaries(self):
        return self.vocab_index, self.inv_vocab_index

    def add_new_term(self, w):
        self.current_index += 1
        self.vocab_index[w] = self.current_index
        self.inv_vocab_index[self.current_index] = w

    def transform(self, text):
        list_tokens = self.tokenizer(text)
        list_tokens = [x for x in list_tokens if x not in self.stop_words]
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


class CustomVectorizer:

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def get_vector_for_tuple(self, tuple):
        vector = self.vectorizer.transform([tuple])
        return vector  # vector may or may not be sparse


def get_sample_from_tokens(tokens, vectorizer):
    return


def tokenize(tuple_str, separator):
    clean_tokens = set()
    tokens = tuple_str.split(separator)
    for t in tokens:
        if len(t) < 3:
            continue
        if re.search('[0-9]', t) is not None:
            continue
        t = t.replace('_', ' ')
        t = t.replace('-', ' ')
        t = t.lower()
        t_tokens = t.split(' ')
        for token in t_tokens:
            clean_tokens.add(token)
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