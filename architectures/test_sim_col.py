from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, Lambda
import keras.backend as K
from keras.optimizers import SGD
import numpy as np
import pickle
from preprocessing.text_processor import IndexVectorizer
from preprocessing import text_processor as tp


demo = False


def main():

    o_path = "/data/eval/qatask/simcol/"

    from utils import prepare_sqa_data
    #data = prepare_sqa_data.get_sqa(filter_stopwords=True)

    pos_dic = prepare_sqa_data.get_pairs_cols()

    true_pairs = []
    for v in pos_dic.values():
        for el1, el2 in v:
            true_pairs.append((el1, el2, 0))

    from random import randint
    num_cols = len(pos_dic)
    num_rows = len(list(pos_dic.values())[0])
    false_pairs = []
    num_false_pairs = len(true_pairs)
    current_false_pairs = 0
    idx_col = dict()
    for idx, el in enumerate(list(pos_dic.keys())):
        idx_col[idx] = el
    seen = set()
    while current_false_pairs < num_false_pairs:
        rc1 = randint(0, num_cols - 1)
        rc2 = randint(0, num_cols - 1)
        rr1 = randint(0, num_rows - 1)
        rr2 = randint(0, num_rows - 1)
        if rc1 == rc2:  # same col are true
            continue
        e1 = pos_dic[idx_col[rc1]][rr1][0]
        e2 = pos_dic[idx_col[rc2]][rr2][0]
        if e1 + e2 not in seen:
            false_pairs.append((e1, e2, 1))
            current_false_pairs += 1
        seen.add(e1 + e2)

    # uns_spos, loc_dic = prepare_sqa_data.get_spo_from_uns()

    # spos = spos + uns_spos

    with open(o_path + "true_pairs.pkl", "wb") as f:
        pickle.dump(true_pairs, f)

    print("True pairs: " + str(len(true_pairs)))

    print("Negative pairs: " + str(len(false_pairs)))

    all_data = true_pairs + false_pairs

    vocab = dict()

    sparsity_code_size = 48

    idx_vectorizer = IndexVectorizer(vocab_index=vocab, sparsity_code_size=sparsity_code_size, tokenizer_sep=" ")
    vectorizer = tp.CustomVectorizer(idx_vectorizer)

    # vectorization happens here
    X1 = []
    X2 = []
    Y = []
    for e1, e2, label in all_data:
        ve1 = vectorizer.get_vector_for_tuple(e1)
        ve1 = ve1.toarray()[0]
        ve2 = vectorizer.get_vector_for_tuple(e2)
        ve2 = ve2.toarray()[0]
        X1.append(ve1)
        X2.append(ve2)
        Y.append(label)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y = np.asarray(Y)

    vocab, inv_vocab = vectorizer.get_vocab_dictionaries()

    # def model1():
    input_dim = sparsity_code_size * 32

    # declare network
    i1 = Input(shape=(input_dim,), name="i1")
    i2 = Input(shape=(input_dim,), name="i2")

    base = Sequential()
    base.add(Dense(1024, input_shape=(input_dim,), activation='relu'))
    base.add(Dense(768, activation='relu'))
    base.add(Dense(512, activation='relu'))
    base.add(Dense(256, activation='relu'))
    base.add(Dense(128, activation='relu'))
    base.add(Dense(64, activation='relu'))

    emb_1 = base(i1)
    emb_2 = base(i2)

    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return shape1[0], 1

    def contrastive_loss(y_true, y_pred):
        margin = 1
        # Correct this to reflect, Y=0 means similar and Y=1 means dissimilar. Think of it as distance
        return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([emb_1, emb_2])

    fullmodel = Model(input=[i1, i2], output=distance)

    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    fullmodel.compile(optimizer=opt, loss=contrastive_loss, metrics=['accuracy'])

    fullmodel.summary()

    def size(model):  # Compute number of params in a model (the actual number of floats)
        return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])

    print("trainable params: " + str(size(fullmodel)))

    fullmodel.fit([X1, X2], Y, epochs=300, shuffle=True, batch_size=32)

    encoder = Model(i1, emb_1)

    fullmodel.save(o_path + "/sim.h5")
    encoder.save(o_path + "/sim_encoder.h5")

    with open(o_path + "tf_dictionary.pkl", "wb") as f:
        pickle.dump(vocab, f)


if __name__ == "__main__":
    print("test sim network")
    main()
