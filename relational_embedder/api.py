import pickle
import time
from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cosine

import word2vec
from relational_embedder import composition
from relational_embedder.data_prep import data_prep_utils as dpu


class SIMF(Enum):
    COSINE = 0
    EUCLIDEAN = 1


class Fabric:

    def __init__(self, row_we_model, col_we_model, row_relational_embedding, col_relational_embedding, path_to_relations):
        self.M_R = row_we_model
        self.M_C = col_we_model
        self.RE_R = row_relational_embedding
        self.RE_C = col_relational_embedding
        self.path_to_relations = path_to_relations

    """
    Basic functions
    """
    def row_vector_for(self, cell=None, attribute=None, table=None):
        vec = None
        if cell:
            cell = dpu.encode_cell(cell)
            vec = self.M_R.get_vector(cell)
        elif table:
            table = dpu.encode_cell(table)
            if attribute:
                attribute = dpu.encode_cell(attribute)
                vec = self.RE_R[table]["columns"][attribute]
            else:
                vec = self.RE_R[table]["vector"]
        elif attribute:
            attribute = dpu.encode_cell(attribute)
            print("Not supported yet!")
            return
        return vec

    def col_vector_for(self, cell=None, attribute=None, table=None):
        vec = None
        if cell:
            cell = dpu.encode_cell(cell)
            vec = self.M_C.get_vector(cell)
        elif table:
            table = dpu.encode_cell(table)
            if attribute:
                attribute = dpu.encode_cell(attribute)
                vec = self.RE_C[table]["columns"][attribute]
            else:
                vec = self.RE_C[table]["vector"]
        elif attribute:
            attribute = dpu.encode_cell(attribute)
            print("Not supported yet!")
            return
        return vec

    def similarity_between_vectors(self, v1, v2, simf=SIMF.COSINE):
        similarity = 0
        if simf == SIMF.COSINE:
            # similarity = np.dot(v1, v2)
            # similarity = self.re_range_score(similarity)
            similarity = 1 - cosine(v1, v2)
        elif simf == SIMF.EUCLIDEAN:
            similarity = 1 - euclidean(v1, v2)
        return similarity

    def similarity_between(self, entity1, entity2, simf=SIMF.COSINE):
        x = dpu.encode_cell(entity1)
        y = dpu.encode_cell(entity2)
        vec_x = self.M_C.get_vector(x)
        vec_y = self.M_C.get_vector(y)
        return self.similarity_between_vectors(vec_x, vec_y, simf=simf)

    def relatedness_between(self, entity1, entity2, simf=SIMF.COSINE):
        x = dpu.encode_cell(entity1)
        y = dpu.encode_cell(entity2)
        vec_x = self.M_R.get_vector(x)
        vec_y = self.M_R.get_vector(y)
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
        indexes, metrics = self.M_R.analogy(pos=[x, y], neg=[z], n=10)
        res = self.M_R.generate_response(indexes, metrics).tolist()
        return res

    """
    Topk functions
    """

    def topk_similar_entities(self, el, k=10, simf=SIMF.COSINE):
        # el = dpu.encode_cell(input_string)
        indexes = []
        metrics = []
        if simf == SIMF.COSINE:
            # indexes, metrics = self.M_C.cosine(el, n=k)
            distances = np.dot(self.M_C.vectors, el.T)
            indexes = np.argsort(distances)[::-1][1:k + 1]
            metrics = distances[indexes]
        elif simf == SIMF.EUCLIDEAN:
            indexes, metrics = self.M_C.euclidean(el, n=k)
        res = self.M_C.generate_response(indexes, metrics).tolist()
        return res

    def topk_related_entities(self, el, k=10, simf=SIMF.COSINE):
        # el = dpu.encode_cell(input_string)
        indexes = []
        metrics = []
        if simf == SIMF.COSINE:
            # indexes, metrics = self.M_R.cosine(el, n=k)
            distances = np.dot(self.M_R.vectors, el.T)
            indexes = np.argsort(distances)[::-1][1:k + 1]
            metrics = distances[indexes]
        elif simf == SIMF.EUCLIDEAN:
            indexes, metrics = self.M_R.euclidean(el, n=k)
        res = self.M_R.generate_response(indexes, metrics).tolist()
        return res

    def topk_related_entities_denoising(self, el, k=10, simf=SIMF.COSINE):
        res = self.topk_related_entities(el, k=k, simf=simf)

        coh_set = defaultdict(int)
        for e, score in res:
            ev = self.M_R.get_vector(e)
            if np.array_equal(el, ev):  # don't include the querying vector
                continue
            sres = self.topk_related_entities(ev, k=10, simf=simf)
            for se, s_score in sres:
                coh_set[se] += 1

        coh_set = {key: (v / k) for key, v in coh_set.items()}

        final_res = sorted(coh_set.items(), key=lambda x: x[1], reverse=True)

        return list(final_res)[:k]

    def topk_related_entities_conditional_denoising(self, el, k=10, simf=SIMF.COSINE):
        res = self.topk_related_entities(el, k=k, simf=simf)
        fixed_group = res[:5]  # top 5 elements
        coh_set = defaultdict(int)
        for e, score in res:
            ev = self.M_R.get_vector(e)
            if np.array_equal(el, ev):  # don't include the querying vector
                continue
            sres = self.topk_related_entities(ev, k=10, simf=simf)
            for se, s_score in sres:
                coh_set[se] += 1

        coh_set = {key: (v / k) for key, v in coh_set.items()}

        # filter fixed_group elements from coh_set
        coh_set = {k: v for k, v in coh_set if k not in fixed_group}

        final_res = sorted(coh_set.items(), key=lambda x: x[1], reverse=True)

        size_to_fill = 5  # fixed for now
        candidate_replacements = len(coh_set)
        if candidate_replacements >= size_to_fill:
            total_replacements = 5
        else:
            total_replacements = size_to_fill - candidate_replacements
        denoised_ranking = fixed_group + res[5:][:(5 - total_replacements)] + final_res[:total_replacements]

        assert(len(denoised_ranking) == k)
        return denoised_ranking

    def topk_similar_relations(self, vec_e, k=None, simf=SIMF.COSINE):
        topk = []
        for vec, relation in self.relation_iterator_c():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                # similarity = np.dot(vec_e, vec)
                # similarity = self.re_range_score(similarity)
                similarity = 1 - cosine(vec_e, vec)
            elif simf == SIMF.EUCLIDEAN:
                similarity = 1 - euclidean(vec_e, vec)
            topk.append((relation, similarity))
        topk = sorted(topk, key=lambda x: x[1], reverse=True)
        if k:
            return topk[:k]
        else:
            return topk

    def topk_related_relations(self, vec_e, k=None, simf=SIMF.COSINE):
        topk = []
        for vec, relation in self.relation_iterator_r():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                # similarity = np.dot(vec_e, vec)
                # similarity = self.re_range_score(similarity)
                similarity = 1 - cosine(vec_e, vec)
            elif simf == SIMF.EUCLIDEAN:
                similarity = 1 - euclidean(vec_e, vec)
            topk.append((relation, similarity))
        topk = sorted(topk, key=lambda x: x[1], reverse=True)
        if k:
            return topk[:k]
        else:
            return topk

    def topk_similar_columns(self, vec_e, k=None, simf=SIMF.COSINE):
        topk = []
        for vec, relation, column in self.column_iterator_c():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                # similarity = np.dot(vec_e, vec)
                # similarity = self.re_range_score(similarity)
                similarity = 1 - cosine(vec_e, vec)
            elif simf == SIMF.EUCLIDEAN:
                similarity = 1 - euclidean(vec_e, vec)
            topk.append((column, relation, similarity))
        topk = sorted(topk, key=lambda x: x[2], reverse=True)
        if k:
            return topk[:k]
        else:
            return topk

    def topk_related_columns(self, vec_e, k=None, simf=SIMF.COSINE):
        topk = []
        for vec, relation, column in self.column_iterator_r():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                # similarity = np.dot(vec_e, vec)
                # similarity = self.re_range_score(similarity)
                similarity = 1 - cosine(vec_e, vec)
            elif simf == SIMF.EUCLIDEAN:
                similarity = 1 - euclidean(vec_e, vec)
            topk.append((column, relation, similarity))
        topk = sorted(topk, key=lambda x: x[2], reverse=True)
        if k:
            return topk[:k]
        else:
            return topk

    def topk_related_rows(self, vec_e, k=5, simf=SIMF.COSINE):
        topk = []
        min_el = -1000
        for vec, relation, row_idx in self.row_iterator_r():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                # similarity = np.dot(vec_e, vec)
                # similarity = self.re_range_score(similarity)
                similarity = 1 - cosine(vec_e, vec)
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
    Explanation API
    """
    def why_similar(self, entity1, entity2):
        return

    def how_do_they_relate(self, entity1, entity2):
        return


    """
    Iterator Utils
    """

    def relation_iterator_r(self):
        """
        Given a relational embedding, iterate over the relation vectors
        :param relational_embedding:
        :return:
        """
        for relation, v in self.RE_R.items():
            yield v["vector"], relation

    def relation_iterator_c(self):
        """
        Given a relational embedding, iterate over the relation vectors
        :param relational_embedding:
        :return:
        """
        for relation, v in self.RE_C.items():
            yield v["vector"], relation

    def column_iterator_r(self):
        """
        Given a relational embedding, iterate over the relation vectors
        :param relational_embedding:
        :return:
        """
        for relation, v in self.RE_R.items():
            for column, vector in self.RE_R[relation]["columns"].items():
                yield vector, relation, column

    def column_iterator_c(self):
        """
        Given a relational embedding, iterate over the relation vectors
        :param relational_embedding:
        :return:
        """
        for relation, v in self.RE_C.items():
            for column, vector in self.RE_C[relation]["columns"].items():
                yield vector, relation, column

    def row_iterator_r(self):
        """
        Given a relational embedding, iterate over the rows
        :param relational_embedding:
        :return:
        """
        for relation, v in self.RE_R.items():
            for row_idx, vector in self.RE_R[relation]["rows"].items():
                yield vector, relation, row_idx

    """
    Experimental
    """
    def concept_qa(self, entity, relation, attribute, n=20, simf=SIMF.COSINE):
        entity = dpu.encode_cell(entity)
        indexes = []
        metrics = []
        if simf == SIMF.COSINE:
            indexes, metrics = self.M_R.cosine(entity, n=n)
        elif simf == SIMF.EUCLIDEAN:
            indexes, metrics = self.M_R.euclidean(entity, n=n)
        res = self.M.generate_response(indexes, metrics).tolist()
        res = [(e, self.re_range_score(score)) for e, score in res]
        vec_attribute = self.RE_R[relation]["columns"][attribute]
        # vec_attribute = self.RE[relation+"."+attribute]
        candidate_attribute_sim = []
        for e, score in res:
            vec_e = self.M_R.get_vector(e)  # no need to normalize e --- it's already normalized
            similarity_to_attr = 0
            if simf == SIMF.COSINE:
                # similarity_to_attr = np.dot(vec_e, vec_attribute)
                # similarity_to_attr = self.re_range_score(similarity_to_attr)
                distance_to_attr = cosine(vec_e, vec_attribute)
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
            vec_e = self.M_R.get_vector(e)
            top_attr = self.topk_related_columns(vec_e, k=ranking_cut, simf=simf)
            keep = False
            for column, relation, similarity in top_attr:
                if column == attribute:
                    keep = True
            if keep:
                denoised_candidate_attr_sim.append((e, sim))
        return denoised_candidate_attr_sim

    def concept_expansion(self, instance, relation, concept, k=5):
        res = []
        concept_vec = self.RE_C[relation]["columns"][concept]
        threshold_sim = self.similarity_between_vectors(concept_vec, self.col_vector_for(cell=instance))
        print(str(threshold_sim))
        top_similar = self.topk_similar_entities(instance, k=k)
        for e, score in top_similar:
            sim = self.similarity_between(concept_vec, self.col_vector_for(cell=e))
            if sim >= threshold_sim:
                res.append(e)
        return res

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

    # def serialize_relemb(self, path):
    #     with open(path, 'wb') as f:
    #         pickle.dump(self.RE, f)
    #     print("Relational Embedding serialized to: " + str(path))


def init(path_to_row_we_model, path_to_col_we_model, path_to_relations):
    st = time.time()
    row_we_model = word2vec.load(path_to_row_we_model)
    col_we_model = word2vec.load(path_to_col_we_model)
    et = time.time()
    we_loading_time = et - st
    st = time.time()
    row_relemb, col_relemb = composition.compose_dataset(path_to_relations, row_we_model, col_we_model)
    et = time.time()
    relemb_build_time = et - st
    api = Fabric(row_we_model, col_we_model, row_relemb, col_relemb, path_to_relations)
    print("Time to load WE model: " + str(we_loading_time))
    print("Time to build relemb: " + str(relemb_build_time))
    return api


def load(path_to_row_we_model, path_to_col_we_model, path_to_row_relemb, path_to_col_relemb, path_to_relations):
    st = time.time()
    row_we_model = word2vec.load(path_to_row_we_model)
    col_we_model = word2vec.load(path_to_col_we_model)
    et = time.time()
    we_loading_time = et - st
    st = time.time()
    with open(path_to_row_relemb, "rb") as f:
        row_relational_embedding = pickle.load(f)
    with open(path_to_col_relemb, "rb") as f:
        col_relational_embedding = pickle.load(f)
    et = time.time()
    relemb_build_time = et - st
    api = Fabric(row_we_model, col_we_model, row_relational_embedding, col_relational_embedding, path_to_relations)
    print("Time to load WE model: " + str(we_loading_time))
    print("Time to build relemb: " + str(relemb_build_time))
    return api


if __name__ == "__main__":
    print("Fabric - relational embedding API")
