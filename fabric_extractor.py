import inputoutput as IO
from collections import OrderedDict
import numpy as np
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

    return trainer


def train(s_in_mat, p_in_mat, o_in_mat, embeddings, config, trainer):
    # Control parameters
    report_interval = 10  # report every 10 epochs

    # Model Hyperparameters
    epochs = 500
    nbatches = 1000
    lr_embeddings = 0.01
    lr_param = 0.01
    batch_size = s_in_mat.shape[1] / nbatches

    for epoch in epochs:
        # Shuffle data
        permutation_idx = np.random.permutation(s_in_mat.shape[1])
        s_in_mat = s_in_mat[:, permutation_idx]
        p_in_mat = p_in_mat[:, permutation_idx]
        o_in_mat = o_in_mat[:, permutation_idx]

        # Get negative samples somehow
        # TODO:
        microbatch_neg_s = []
        microbatch_neg_o = []

        for i in range(nbatches):
            microbatch_s = s_in_mat[:, i + batch_size: (i + 1) + batch_size]
            microbatch_p = p_in_mat[:, i + batch_size: (i + 1) + batch_size]
            microbatch_o = o_in_mat[:, i + batch_size: (i + 1) + batch_size]
            iteration_output = trainer(lr_embeddings, lr_param, microbatch_s,
                                       microbatch_o, microbatch_p, microbatch_neg_s,
                                       microbatch_neg_o)
            avg_cost, ratio_updates = iteration_output
            embeddings.normalize()
        if epoch % report_interval == 0:
            continue



if __name__ == "__main__":
    print("Extract (learn) fabric from input data")
