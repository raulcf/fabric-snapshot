import word2vec
from data_prep import data_prep_utils as dpu
from relational_embedder import composition
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import heapq


class Fabric:

    def __init__(self, we_model, relational_embedding, path_to_relations):
        self.M = we_model
        self.RE = relational_embedding
        self.path_to_relations = path_to_relations

    def concept_qa(self, entity, relation, attribute, n=20):
        entity = dpu.encode_cell(entity)
        indexes, metrics = self.M.cosine(entity, n=n)
        res = self.M.generate_response(indexes, metrics).tolist()
        vec_attribute = self.RE[relation]["columns"][attribute]
        candidate_attribute_sim = []
        for e, score in res:
            vec_e = self.M.get_vector(e)  # no need to normalize e --- it's already normalized
            distance = cosine(vec_e, vec_attribute)
            similarity = 1 - distance
            candidate_attribute_sim.append((e, similarity))
        candidate_attribute_sim = sorted(candidate_attribute_sim, key=lambda x: x[1], reverse=True)
        return candidate_attribute_sim

    def concept_qa_denoising(self, entity, relation, attribute, n=20, denoise_heuristic=3):
        entity = dpu.encode_cell(entity)
        indexes, metrics = self.M.cosine(entity, n=n)
        res = self.M.generate_response(indexes, metrics).tolist()
        vec_attribute = self.RE[relation]["columns"][attribute]
        candidate_attribute_sim = []
        for e, score in res:
            vec_e = self.M.get_vector(e)  # no need to normalize e --- it's already normalized
            distance = cosine(vec_e, vec_attribute)
            similarity = 1 - distance
            candidate_attribute_sim.append((e, similarity))
        candidate_attribute_sim = sorted(candidate_attribute_sim, key=lambda x: x[1], reverse=True)
        # now we have a list of candidates, denoise the ranking by checking that each is also closer to the attr at hand
        ranking_cut = denoise_heuristic
        denoised_candidate_attr_sim = []
        for e, sim in candidate_attribute_sim:
            vec_e = self.M.get_vector(e)
            top_attr = self.topk_columns(vec_e, k=ranking_cut)
            keep = False 
            for column, relation, similarity in top_attr:
                #if column != attribute or relation != relation:
                if column == attribute: 
                    keep = True
            if keep:
                denoised_candidate_attr_sim.append((e, sim))
        return denoised_candidate_attr_sim 

    def entity_to_attribute(self, entities, n=2):
        res = []
        for entity in entities:
            entity = dpu.encode_cell(entity)
            vec_e = self.M.get_vector(entity)
            topk = self.topk_columns(vec_e, k=n)
            res.append((entity, topk))
        return res

    def concept_expansion(self, instance, relation, concept, k=5):
        res = []
        concept_vec = self.RE[relation]["columns"][concept]
        threshold_sim = self.similarity_between_vectors(concept_vec, self.vector_for_entity(cell=instance))
        print(str(threshold_sim))
        top_similar = self.topk_similar_vectors(instance, k=k)
        for e, score in top_similar:
            sim = self.similarity_between(concept_vec, self.vector_for_entity(cell=e))
            if sim >= threshold_sim:
                res.append(e)
        return res

    def similarity_between_vectors(self, v1, v2):
        distance = cosine(v1, v2)
        similarity = 1 - distance
        return similarity
        
    def similarity_between(self, entity1, entity2):
        x = dpu.encode_cell(entity1)
        y = dpu.encode_cell(entity2)
        vec_x = self.M.get_vector(x)
        vec_y = self.M.get_vector(y)
        return self.similarity_between_vectors(vec_x, vec_y)

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

    """
    Topk functions
    """

    def topk_similar_vectors(self, input_string, k=10):
        el = dpu.encode_cell(input_string)
        indexes, metrics = self.M.cosine(el, n=k)
        res = self.M.generate_response(indexes, metrics).tolist()
        return res

    def topk_relations(self, vec_e, k=None):
        topk = []
        for vec, relation in self.relation_iterator():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            distance = cosine(vec_e, vec)
            similarity = 1 - distance
            topk.append((relation, similarity))
        topk = sorted(topk, key=lambda x: x[1], reverse=True)
        if k:
            return topk[:k]
        else:
            return topk

    def topk_columns(self, vec_e, k=None):
        topk = []
        for vec, relation, column in self.column_iterator():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            distance = cosine(vec_e, vec)
            similarity = 1 - distance
            topk.append((column, relation, similarity))
        topk = sorted(topk, key=lambda x: x[2], reverse=True)
        if k:
            return topk[:k]
        else:
            return topk

    def topk_rows(self, vec_e, k=5):
        # class HeapObj:
        #     def __init__(self, row, relation, similarity):
        #         self.row = row
        #         self.relation = relation
        #         self.similarity = similarity
        #
        #     def __lt__(self, other):
        #         return self.similarity < other.similarity
        # topk = heapq.heapify([])
        topk = []
        min_el = -1000
        for vec, relation, row_idx in self.row_iterator():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            distance = cosine(vec_e, vec)
            similarity = 1 - distance
            # decide if we keep it or not
            if similarity > min_el:
                #row = self.resolve_row_idx(row_idx, relation)
                # Add and keep fixed-size
                topk.append((row_idx, relation, similarity))
                topk = sorted(topk, key=lambda x: x[2], reverse=True)
                topk = topk[:k]
                min_el = topk[-1][2]  # update min el to last value in list
        # Once found the row_idx, resolve them to actual rows before returning
        to_return = []
        for row_idx, relation, similarity in topk:
            row = self.resolve_row_idx(row_idx, relation)
            to_return.append((row, relation, similarity))
        return to_return

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
