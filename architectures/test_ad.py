
from __future__ import print_function

import numpy as np
import pickle
import time

from preprocessing.text_processor import IndexVectorizer
from preprocessing import text_processor as tp


demo = False


def main():

    from utils import prepare_sqa_data

    spos = prepare_sqa_data.get_spo_from_rel(filter_stopwords=True)

    uns_spos, loc_dic = prepare_sqa_data.get_spo_from_uns()

    spos = spos + uns_spos

    pos_samples = []

    # positive pairs
    for s, p, o in spos:
        pos_samples.append(s + " " + p)
        pos_samples.append(s + " " + o)
        pos_samples.append(p + " " + o)

    all_data = pos_samples

    vocab = dict()

    sparsity_code_size = 48

    idx_vectorizer = IndexVectorizer(vocab_index=vocab, sparsity_code_size=sparsity_code_size, tokenizer_sep=" ")
    vectorizer = tp.CustomVectorizer(idx_vectorizer)

    # vectorization happens here
    X = []
    for el in all_data:
        ve = vectorizer.get_vector_for_tuple(el)
        ve = ve.toarray()[0]
        X.append(ve)

    X = np.asarray(X)

    vocab, inv_vocab = vectorizer.get_vocab_dictionaries()

    # def model1():
    input_dim = sparsity_code_size * 32

    from architectures import fabric_binary as bae
    model = bae.declare_model(input_dim, 256)
    model = bae.compile_model(model)

    st = time.time()

    model.fit(X, X, epochs=250, batch_size=16, shuffle=True)

    et = time.time()
    print("Total time: " + str(et - st))

    o_path = "/data/eval/qatask/ad/"

    bae.save_model_to_path(model, o_path, log="ad")

    with open(o_path + "tf_dictionary.pkl", "wb") as f:
        pickle.dump(vocab, f)


if __name__ == "__main__":
    print("test anomaly detector")
    main()
