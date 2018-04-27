import pickle
import time
from enum import Enum

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

import word2vec
from relational_embedder import composition
from relational_embedder.data_prep import data_prep_utils as dpu


class SIMF(Enum):
    COSINE = 0
    EUCLIDEAN = 1


class Fabric:

    def __init__(self, we_model, relational_embedding, path_to_relations):
        self.M = we_model
        self.RE = relational_embedding
        self.path_to_relations = path_to_relations

    """
    Representation functions
    """
    def find_instance_representation(self, instance):
        # TODO:
        return

    def find_attribute_representation(self, attribute):
        # TODO:
        return

    """
    Main function API
    """

    def _concept_qa_no_avg_rerank(self, entity, relation, attribute, n=20, simf=SIMF.COSINE):
        entity = dpu.encode_cell(entity)
        indexes = []
        metrics = []
        if simf == SIMF.COSINE:
            indexes, metrics = self.M.cosine(entity, n=n)
        elif simf == SIMF.EUCLIDEAN:
            indexes, metrics = self.M.euclidean(entity, n=n)
        res = self.M.generate_response(indexes, metrics).tolist()
        vec_attribute = self.RE[relation]["columns"][attribute]
        # vec_attribute = self.RE[relation+"."+attribute]
        candidate_attribute_sim = []
        for e, score in res:
            vec_e = self.M.get_vector(e)  # no need to normalize e --- it's already normalized
            similarity = 0
            if simf == SIMF.COSINE:
                similarity = np.dot(vec_e, vec_attribute)
                similarity = self.re_range_score(similarity)
            elif simf == SIMF.EUCLIDEAN:
                similarity = 1 - euclidean(vec_e, vec_attribute)
            candidate_attribute_sim.append((e, similarity))
        candidate_attribute_sim = sorted(candidate_attribute_sim, key=lambda x: x[1], reverse=True)
        return candidate_attribute_sim

    def concept_qa(self, entity, relation, attribute, n=20, simf=SIMF.COSINE):
        entity = dpu.encode_cell(entity)
        indexes = []
        metrics = []
        if simf == SIMF.COSINE:
            indexes, metrics = self.M.cosine(entity, n=n)
        elif simf == SIMF.EUCLIDEAN:
            indexes, metrics = self.M.euclidean(entity, n=n)
        res = self.M.generate_response(indexes, metrics).tolist()
        res = [(e, self.re_range_score(score)) for e, score in res]
        vec_attribute = self.RE[relation]["columns"][attribute]
        # vec_attribute = self.RE[relation+"."+attribute]
        candidate_attribute_sim = []
        for e, score in res:
            vec_e = self.M.get_vector(e)  # no need to normalize e --- it's already normalized
            similarity_to_attr = 0
            if simf == SIMF.COSINE:
                similarity_to_attr = np.dot(vec_e, vec_attribute)
                similarity_to_attr = self.re_range_score(similarity_to_attr)
                # distance_to_attr = cosine(vec_e, vec_attribute)
            elif simf == SIMF.EUCLIDEAN:
                similarity_to_attr = 1 - euclidean(vec_e, vec_attribute)
            # avg distance between original entity to each ranking entity and each ranking entity and target attr
            similarity = (similarity_to_attr + score) / 2
            candidate_attribute_sim.append((e, similarity))
        candidate_attribute_sim = sorted(candidate_attribute_sim, key=lambda x: x[1], reverse=True)
        return candidate_attribute_sim

    def concept_qa_denoising(self, entity, relation, attribute, n=20, denoise_heuristic=3, simf=SIMF.COSINE):
        candidate_attribute_sim = self.concept_qa(entity, relation, attribute, n=n, simf=simf)
        # now we have a list of candidates, denoise the ranking by checking that each is also closer to the attr at hand
        ranking_cut = denoise_heuristic
        denoised_candidate_attr_sim = []
        for e, sim in candidate_attribute_sim:
            vec_e = self.M.get_vector(e)
            top_attr = self.topk_columns(vec_e, k=ranking_cut, simf=simf)
            keep = False
            for column, relation, similarity in top_attr:
                if column == attribute:
                    keep = True
            if keep:
                denoised_candidate_attr_sim.append((e, sim))
        return denoised_candidate_attr_sim

    @DeprecationWarning
    def entity_to_attribute(self, entities, n=2, simf=SIMF.COSINE):
        res = []
        for entity in entities:
            entity = dpu.encode_cell(entity)
            vec_e = self.M.get_vector(entity)
            topk = self.topk_columns(vec_e, k=n, simf=simf)
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

    def similarity_between_vectors(self, v1, v2, simf=SIMF.COSINE):
        similarity = 0
        if simf == SIMF.COSINE:
            similarity = np.dot(v1, v2)
            similarity = self.re_range_score(similarity)
        elif simf == SIMF.EUCLIDEAN:
            similarity = 1 - euclidean(v1, v2)
        return similarity

    def similarity_between(self, entity1, entity2, simf=SIMF.COSINE):
        x = dpu.encode_cell(entity1)
        y = dpu.encode_cell(entity2)
        vec_x = self.M.get_vector(x)
        vec_y = self.M.get_vector(y)
        return self.similarity_between_vectors(vec_x, vec_y, simf=simf)

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

    def topk_similar_vectors(self, input_string, k=10, simf=SIMF.COSINE):
        el = dpu.encode_cell(input_string)
        indexes = []
        metrics = []
        if simf == SIMF.COSINE:
            indexes, metrics = self.M.cosine(el, n=k)
        elif simf == SIMF.EUCLIDEAN:
            indexes, metrics = self.M.euclidean(el, n=k)
        res = self.M.generate_response(indexes, metrics).tolist()
        return res

    def topk_relations(self, vec_e, k=None, simf=SIMF.COSINE):
        topk = []
        for vec, relation in self.relation_iterator():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                similarity = np.dot(vec_e, vec)
                similarity = self.re_range_score(similarity)
            elif simf == SIMF.EUCLIDEAN:
                similarity = 1 - euclidean(vec_e, vec)
            topk.append((relation, similarity))
        topk = sorted(topk, key=lambda x: x[1], reverse=True)
        if k:
            return topk[:k]
        else:
            return topk

    def topk_columns(self, vec_e, k=None, simf=SIMF.COSINE):
        topk = []
        for vec, relation, column in self.column_iterator():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                similarity = np.dot(vec_e, vec)
                similarity = self.re_range_score(similarity)
            elif simf == SIMF.EUCLIDEAN:
                similarity = 1 - euclidean(vec_e, vec)
            topk.append((column, relation, similarity))
        topk = sorted(topk, key=lambda x: x[2], reverse=True)
        if k:
            return topk[:k]
        else:
            return topk

    def topk_rows(self, vec_e, k=5, simf=SIMF.COSINE):
        topk = []
        min_el = -1000
        for vec, relation, row_idx in self.row_iterator():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                similarity = np.dot(vec_e, vec)
                similarity = self.re_range_score(similarity)
            elif simf == SIMF.EUCLIDEAN:
                similarity = euclidean(vec_e, vec)
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

    def re_range_score(self, score):
        """
        Given a score in the range [-1, 1], it transforms it to [0,1]
        :param score:
        :return:
        """
        new_value = (score + 1) / 2
        return new_value

    def resolve_row_idx(self, row_idx, relation):
        df = pd.read_csv(self.path_to_relations + "/" + relation, encoding='latin1')
        row = df.iloc[row_idx]
        return row

    def serialize_relemb(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.RE, f)
        print("Relational Embedding serialized to: " + str(path))


def init(path_to_we_model, path_to_relations):
    st = time.time()
    we_model = word2vec.load(path_to_we_model)
    et = time.time()
    we_loading_time = et - st
    st = time.time()
    relational_embedding = composition.compose_dataset(path_to_relations, we_model)
    et = time.time()
    relemb_build_time = et - st
    api = Fabric(we_model, relational_embedding, path_to_relations)
    print("Time to load WE model: " + str(we_loading_time))
    print("Time to build relemb: " + str(relemb_build_time))
    return api


def load(path_to_we_model, path_to_relemb, path_to_relations):
    st = time.time()
    we_model = word2vec.load(path_to_we_model)
    et = time.time()
    we_loading_time = et - st
    st = time.time()
    with open(path_to_relemb, "rb") as f:
        relational_embedding = pickle.load(f)
    et = time.time()
    relemb_build_time = et - st
    api = Fabric(we_model, relational_embedding, path_to_relations)
    print("Time to load WE model: " + str(we_loading_time))
    print("Time to build relemb: " + str(relemb_build_time))
    return api


if __name__ == "__main__":
    print("Fabric - relational embedding API")
