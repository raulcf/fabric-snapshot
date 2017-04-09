import re
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import utils_pre


class CustomVectorizer:

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def get_vector_for_tuple(self, tuple):
        sparse_array = self.vectorizer.transform([tuple])
        return sparse_array


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