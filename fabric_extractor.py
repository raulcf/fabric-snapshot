import inputoutput as IO
from collections import OrderedDict
import numpy as np
from scipy import sparse
import time
import theano
import theano.tensor as T
import theano.sparse as S


def load_input_data(path):
    s_in_mat, p_in_mat, o_in_mat = IO.load_input_matrices(path)
    config = IO.load_config(path)
    num_predicates = config['num_predicates']
    p_in_mat = p_in_mat[-num_predicates:, :]
    return s_in_mat, p_in_mat, o_in_mat, config


def l2_norm(x, y):
    return - T.sqrt(T.sum(T.sqr(x - y), axis=1))


class LayerTrans(object):
    """ from glorotxa/SME
    Class for a layer with two input vectors that performs the sum of
    of the 'left member' and 'right member'i.e. translating x by y.
    """

    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x+y


class Unstructured(object):
    """ from glorotxa/SME
    Class for a layer with two input vectors that performs the linear operator
    of the 'left member'.

    :note: The 'right' member is the relation, therefore this class allows to
    define an unstructured layer (no effect of the relation) in the same
    framework.
    """

    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x


class Embeddings(object):
    """ from glorotxa/SME
    Class for the embeddings
    matrix.
    """

    def __init__(self, rng, N, D, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param N: number of entities, relations or both.
        :param D: dimension of the embeddings.
        :param tag: name of the embeddings for parameter declaration.
        """
        self.N = N
        self.D = D
        wbound = np.sqrt(6. / D)
        W_values = rng.uniform(low=-wbound, high=wbound, size=(D, N))
        W_values = W_values / np.sqrt(np.sum(W_values ** 2, axis=0))
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E' + tag)
        # Define a normalization function with respect to the L_2 norm of the
        # embedding vectors.
        self.updates = OrderedDict({self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=0))})
        self.normalize = theano.function([], [], updates=self.updates)


def FilteredRankingScoreIdx(sl, sr, idxl, idxr, idxo, true_triples):
    """ from glorotxa/SME
    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.

    :param sl: Theano function created with RankLeftFnIdx().
    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errl = []
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        il = np.argwhere(true_triples[:, 0] == l).reshape(-1, )
        io = np.argwhere(true_triples[:, 1] == o).reshape(-1, )
        ir = np.argwhere(true_triples[:, 2] == r).reshape(-1, )

        inter_l = [i for i in ir if i in io]
        rmv_idx_l = [true_triples[i, 0] for i in inter_l if true_triples[i, 0] != l]
        scores_l = (sl(r, o)[0]).flatten()
        scores_l[rmv_idx_l] = -np.inf
        errl += [np.argsort(np.argsort(-scores_l)).flatten()[l] + 1]

        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i, 2] for i in inter_r if true_triples[i, 2] != r]
        scores_r = (sr(l, o)[0]).flatten()
        scores_r[rmv_idx_r] = -np.inf
        errr += [np.argsort(np.argsort(-scores_r)).flatten()[r] + 1]
    return errl, errr


def TrainFn1Member(fnsim, embeddings, leftop, rightop, marge=1.0):
    """
    This function returns a theano function to perform a training iteration,
    contrasting positive and negative triplets. members are given as sparse
    matrices. For one positive triplet there are two or three (if rel == True)
    negative triplets. To create a negative triplet we replace only one member
    at a time.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge for the cost function.
    :param rel: boolean, if true we also contrast w.r.t. a negative relation
                member.
    """

    def margincost(pos, neg, marge=1.0):
        out = neg - pos + marge
        return T.sum(out * (out > 0)), out > 0

    embedding, relationl, relationr = embeddings

    # Inputs
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')

    # Graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    # Negative 'left' member
    similn = fnsim(leftop(lhsn, rell), rightop(rhs, relr))
    # Negative 'right' member
    simirn = fnsim(leftop(lhs, rell), rightop(rhsn, relr))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)
    cost = costl + costr
    out = T.concatenate([outl, outr])
    # List of inputs of the function
    list_in = [lrembeddings, lrparams, inpl, inpr, inpo, inpln, inprn]

    if hasattr(fnsim, 'params'):
        # If the similarity function has some parameters, we update them too.
        gradientsparams = T.grad(cost, leftop.params + rightop.params + fnsim.params)
        updates = OrderedDict((i, i - lrparams * j) for i, j in zip(leftop.params + rightop.params + fnsim.params, gradientsparams))
    else:
        gradientsparams = T.grad(cost, leftop.params + rightop.params)
        updates = OrderedDict((i, i - lrparams * j) for i, j in zip(leftop.params + rightop.params, gradientsparams))
    gradients_embedding = T.grad(cost, embedding.E)
    newE = embedding.E - lrembeddings * gradients_embedding
    updates.update({embedding.E: newE})
    if type(embeddings) == list:
        # If there are different embeddings for the relation member.
        gradients_embedding = T.grad(cost, relationl.E)
        newE = relationl.E - lrparams * gradients_embedding
        updates.update({relationl.E: newE})
        gradients_embedding = T.grad(cost, relationr.E)
        newE = relationr.E - lrparams * gradients_embedding
        updates.update({relationr.E: newE})
    """
    Theano function inputs.
    :input lrembeddings: learning rate for the embeddings.
    :input lrparams: learning rate for the parameters.
    :input inpl: sparse csr matrix representing the indexes of the positive
                 triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive
                 triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive
                 triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative
                  triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative
                  triplet 'right' member, shape=(#examples,N [Embeddings]).
    :opt input inpon: sparse csr matrix representing the indexes of the
                      negative triplet relation member, shape=(#examples,N
                      [Embeddings]).

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """
    return theano.function(list_in, [T.mean(cost), T.mean(out)], updates=updates, on_unused_input='ignore')


def RankLeftFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = embeddings

    # Inputs
    idxr = T.iscalar('idxr')
    idxo = T.iscalar('idxo')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    tmp = rightop(rhs, relr)
    simi = fnsim(leftop(lhs, rell), tmp.reshape((1, tmp.shape[1])))
    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo], [simi], on_unused_input='ignore')


def RankRightFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = embeddings

    # Inputs
    idxl = T.iscalar('idxl')
    idxo = T.iscalar('idxo')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    tmp = leftop(lhs, rell)
    simi = fnsim(tmp.reshape((1, tmp.shape[1])), rightop(rhs, relr))
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxo], [simi], on_unused_input='ignore')


def declare_TransE_model(config):

    num_elements = config['num_elements']
    num_dimensions = 50
    marge = 0.5

    s_op = LayerTrans()
    o_op = Unstructured()
    entity_emb = Embeddings(np.random, num_elements, num_dimensions, 'ENTITY_EMB')
    predic_emb = Embeddings(np.random, num_elements, num_dimensions, 'PREDICATE_EMB')
    embeddings = [entity_emb, predic_emb, predic_emb]

    trainer = TrainFn1Member(l2_norm, embeddings, s_op, o_op, marge=marge)

    total_entities = config['unique_s'] + config['unique_o'] + config['unique_so']

    ranker_s = RankLeftFnIdx(l2_norm, embeddings, s_op, o_op, subtensorspec=total_entities)

    ranker_o = RankRightFnIdx(l2_norm, embeddings, s_op, o_op,subtensorspec=total_entities)

    return trainer, embeddings, ranker_s, ranker_o


def create_negative_samples(row_dim, col_dim, range_entities):
    range_entities = range_entities[np.random.permutation(len(range_entities))]
    neg_matrix = sparse.lil_matrix((row_dim, col_dim), dtype=theano.config.floatX)
    entity_idx = 0
    for triple in range(col_dim):
        # grab a random entity and place it for the triple 'triple'
        if entity_idx == len(range_entities):
            # Reset if we run out of tuples
            entity_idx = 0
        neg_matrix[range_entities[entity_idx], triple] = 1
        entity_idx += 1
    return neg_matrix.tocsr()


def train(train, eval_train, validation, eval_validation, test, eval_test,
          all_data, embeddings, config,
          trainer, ranker_s, ranker_o):
    # Control parameters
    report_interval = 10  # report every 10 epochs

    # Model Hyperparameters
    epochs = 500
    nbatches = 1000
    lr_embeddings = 0.01
    lr_param = 0.01

    # Unpack data
    s_in_mat, p_in_mat, o_in_mat = train
    s_val_mat, p_val_mat, o_val_mat = eval_validation
    s_train_mat, p_train_mat, o_train_mat = eval_train
    s_test_mat, p_test_mat, o_test_mat = eval_test

    batch_size = s_in_mat.shape[1] / nbatches

    cost_array = []
    ratio_updates_array = []

    best_test_error = -1
    best_validation_error = -1
    best_training_error = -1

    for epoch in epochs:
        start_epoch_time = time.time()
        # Shuffle data
        permutation_idx = np.random.permutation(s_in_mat.shape[1])
        s_in_mat = s_in_mat[:, permutation_idx]
        p_in_mat = p_in_mat[:, permutation_idx]
        o_in_mat = o_in_mat[:, permutation_idx]

        # Get negative samples somehow
        total_entities = config['unique_s'] + config['unique_o'] + config['unique_so']
        microbatch_neg_s = create_negative_samples(s_in_mat.shape[0], s_in_mat.shape[1], total_entities)
        microbatch_neg_o = create_negative_samples(s_in_mat.shape[0], s_in_mat.shape[1], total_entities)

        for i in range(nbatches):
            microbatch_s = s_in_mat[:, i + batch_size: (i + 1) + batch_size]
            microbatch_p = p_in_mat[:, i + batch_size: (i + 1) + batch_size]
            microbatch_o = o_in_mat[:, i + batch_size: (i + 1) + batch_size]
            iteration_output = trainer(lr_embeddings, lr_param, microbatch_s,
                                       microbatch_o, microbatch_p, microbatch_neg_s,
                                       microbatch_neg_o)
            avg_cost, ratio_updates = iteration_output
            cost_array += avg_cost / float(batch_size)  # normalize per sample
            ratio_updates_array += ratio_updates
            embeddings.normalize()

        if epoch % report_interval == 0:
            start_test_time = time.time()
            print("EPOCH - " + str(epoch))
            elapsed_time = time.time() - start_epoch_time / report_interval
            print("Avg Epoch running time: " + str(elapsed_time))

            # Checking validation error
            error_s, error_o = FilteredRankingScoreIdx(ranker_s, ranker_o, s_val_mat, o_val_mat, p_val_mat, all_data)
            validation_error = np.mean(error_s, error_o)

            # Checking train error
            error_s, error_o = FilteredRankingScoreIdx(ranker_s, ranker_o, s_train_mat, o_train_mat, p_train_mat, all_data)
            validation_train_error = np.mean(error_s, error_o)

            if best_test_error == -1 or validation_error < best_test_error:
                # Checking test error
                error_s, error_o = FilteredRankingScoreIdx(ranker_s, ranker_o, s_test_mat, o_test_mat, p_test_mat, all_data)
                test_error = np.mean(error_s, error_o)

                best_test_error = test_error
                best_validation_error = validation_error
                best_train_error = validation_train_error

                total_test_time = time.time() - start_test_time

                # Save this model TODO:
                # TODO:
            # Save current model, regardless its quality
            # TODO:


def mat2idx(matrix):
    rows, cols = matrix.nonzero()
    return rows[np.argsort(cols)]


if __name__ == "__main__":
    print("Extract (learn) fabric from input data")

    n_samp = 1000

    s_in_mat, p_in_mat, o_in_mat, config = load_input_data("data/imdb/train")
    s_val_mat, p_val_mat, o_val_mat, config = load_input_data("data/imdb/validation")
    s_test_mat, p_test_mat, o_test_mat, config = load_input_data("data/imdb/test")

    train = (s_in_mat, p_in_mat, o_in_mat)
    validation = (s_val_mat, p_val_mat, o_val_mat)
    test = (s_test_mat, p_test_mat, o_test_mat)

    eval_train = mat2idx(s_in_mat)[:n_samp], mat2idx(p_in_mat)[:n_samp], mat2idx(o_in_mat)[:n_samp]
    eval_validation = mat2idx(s_val_mat)[:n_samp], mat2idx(p_val_mat)[:n_samp], mat2idx(o_val_mat)[:n_samp]
    eval_test = mat2idx(s_test_mat)[:n_samp], mat2idx(p_test_mat)[:n_samp], mat2idx(o_test_mat)[:n_samp]

    s_train_idx, p_train_idx, o_train_idx = mat2idx(s_in_mat), mat2idx(p_in_mat), mat2idx(o_in_mat)
    s_val_idx, p_val_idx, o_val_idx = mat2idx(s_val_mat), mat2idx(p_val_mat), mat2idx(o_val_mat)
    s_test_idx, p_test_idx, o_test_idx = mat2idx(s_test_mat), mat2idx(p_test_mat), mat2idx(o_test_mat)

    all_idx = np.concatenate([s_train_idx, s_val_idx, s_test_idx,
                              p_train_idx, p_val_idx, p_test_idx,
                              o_train_idx, o_val_idx, o_test_idx])

    all_input_data_idx = all_idx.reshape(3, s_train_idx.shape[0] + s_val_idx.shape[0] + s_test_idx.shape[0]).T

    trainer, embeddings, ranker_s, ranker_o = declare_TransE_model(config)

    train(train, eval_train, validation, eval_validation, test, eval_test,
          all_input_data_idx, embeddings, config, trainer, ranker_s, ranker_o)