import word2vec
from data_prep import data_prep_utils as dpu
from relational_embedder import composition
from scipy.spatial.distance import cosine
import pandas as pd


class Fabric:

    def __init__(self, we_model, relational_embedding, path_to_relations):
        self.M = we_model
        self.RE = relational_embedding
        self.path_to_relations = path_to_relations

    def topk_similar_vectors(self, input_string, k=10):
        el = dpu.encode_cell(input_string)
        indexes, metrics = self.M.we_model.cosine(el, n=k)
        res = self.M.generate_response(indexes, metrics).tolist()
        return res

    def analogy(self, x, y, z):
        """
        y is to ??? what z is to x
        :param x:
        :param y:
        :param z:
        :return:
        """
        x = dpu.encode_cell(x)
        y = dpu.encode_cell(y)
        z = dpu.encode_cell(z)
        indexes, metrics = self.M.analogy(pos=[x, y], neg=[z], n=10)
        res = self.M.generate_response(indexes, metrics).tolist()
        return res

    def vector_for_entity(self, cell=None, attribute=None, table=None):
        vec = None
        if cell:
            cell = dpu.encode_cell(cell)
            vec = self.M.get_vector(cell)
        elif table:
            table = dpu.encode_cell(table)
            if attribute:
                attribute = dpu.encode_cell(attribute)
                vec = self.RE[table]["columns"][attribute]
            else:
                vec = self.RE[table]["vector"]
        elif attribute:
            attribute = dpu.encode_cell(attribute)
            print("Not supported yet!")
            return
        return vec

    def topk_relations(self, vec_e, k=None):
        topk = []
        for vec, relation in self.relation_iterator():
            distance = cosine(vec_e, vec)
            similarity = 1 - distance
            topk.append((relation, similarity))
        if k:
            return topk[:k]
        else:
            return topk

    def topk_columns(self, vec_e, k=None):
        topk = []
        for vec, relation, column in self.column_iterator():
            distance = cosine(vec_e, vec)
            similarity = 1 - distance
            topk.append((column, relation, similarity))
        if k:
            return topk[:k]
        else:
            return topk

    def topk_rows(self, vec_e, k=None):
        topk = []
        for vec, relation, row_idx in self.row_iterator():
            distance = cosine(vec_e, vec)
            similarity = 1 - distance
            row = self.resolve_row_idx(row_idx, relation)
            topk.append((row, relation, similarity))
        if k:
            return topk[:k]
        else:
            return topk

    """
    Iterator Utils
    """

    def relation_iterator(self):
        """
        Given a relational embedding, iterate over the relation vectors
        :param relational_embedding:
        :return:
        """
        for relation, v in self.RE.items():
            yield v["vector"], relation

    def column_iterator(self):
        """
        Given a relational embedding, iterate over the relation vectors
        :param relational_embedding:
        :return:
        """
        for relation, v in self.RE.items():
            for column, vector in self.RE[relation]["columns"].items():
                yield vector, relation, column

    def row_iterator(self):
        """
        Given a relational embedding, iterate over the rows
        :param relational_embedding:
        :return:
        """
        for relation, v in self.RE.items():
            for row_idx, vector in self.RE[relation]["rows"].items():
                yield vector, relation, row_idx

    """
    Utils
    """
    def resolve_row_idx(self, row_idx, relation):
        df = pd.read_csv(self.path_to_relations + "/" + relation, encoding='latin1')
        row = df.iloc[row_idx]
        return row


def init(path_to_we_model, path_to_relations):
    we_model = word2vec.load(path_to_we_model)
    relational_embedding = composition.compose_dataset(path_to_relations, we_model)
    api = Fabric(we_model, relational_embedding, path_to_relations)
    return api



if __name__ == "__main__":
    print("Fabric - relational embedding API")
