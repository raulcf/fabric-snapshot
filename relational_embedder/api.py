import pickle
import time
from enum import Enum
from collections import defaultdict

import numpy as np
from numpy import ma
import pandas as pd
from scipy.spatial.distance import euclidean, cosine
from sklearn.cluster import KMeans

import word2vec
from relational_embedder import composition
from relational_embedder.data_prep import data_prep_utils as dpu


class SIMF(Enum):
    COSINE = 0
    EUCLIDEAN = 1


class Fabric:

    def __init__(self, row_we_model, col_we_model, row_relational_embedding,
                 col_relational_embedding, path_to_relations, word_hubness):
        self.M_R = row_we_model
        self.M_C = col_we_model
        self.RE_R = row_relational_embedding
        self.RE_C = col_relational_embedding
        self.path_to_relations = path_to_relations
        self.word_hubness = word_hubness

    """
    text to vector API
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

    """
    combination API
    """

    def combine(self, vecs_to_combine):
        """
        Given a list of vectors, it combines them
        :param vecs_to_combine:
        :return:
        """
        # TODO: probably want to filter out vecs based on hubness?
        vecs_to_combine = np.asarray(vecs_to_combine)
        comb = np.mean(vecs_to_combine, axis=0)
        return comb

    """
    Topk similarity and relatedness API
    """

    def more_entities_like(self, el, k=10, simf=SIMF.COSINE):
        if type(el) is str:
            el = self.col_vector_for(cell=el)
        indexes = []
        metrics = []
        if simf == SIMF.COSINE:
            sims = np.dot(self.M_C.vectors, el.T)
            indexes = np.argsort(sims)[::-1][1:k + 1]
            metrics = sims[indexes]
        elif simf == SIMF.EUCLIDEAN:
            indexes, metrics = self.M_C.euclidean(el, n=k)
        res = self.M_C.generate_response(indexes, metrics).tolist()
        return res

    def topk_related_entities(self, el, k=10, simf=SIMF.COSINE):
        if type(el) is str:
            el = self.row_vector_for(cell=el)
        indexes = []
        metrics = []
        if simf == SIMF.COSINE:
            sims = np.dot(self.M_R.vectors, el.T)
            indexes = np.argsort(sims)[::-1][1:k + 1]
            metrics = sims[indexes]
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
        fixed_group = [e for e, _ in res[:5]]  # top 5 elements
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
        coh_set = {k: v for k, v in coh_set.items() if k not in fixed_group and not np.array_equal(el, self.M_R.get_vector(k))}

        final_res = sorted(coh_set.items(), key=lambda x: x[1], reverse=True)

        size_to_fill = 5  # fixed for now
        candidate_replacements = len(coh_set)
        if candidate_replacements >= size_to_fill:
            total_replacements = 5
        else:
            total_replacements = candidate_replacements
        denoised_ranking = res[:5] + res[5:][:(size_to_fill - total_replacements)] + final_res[:total_replacements]

        assert(len(denoised_ranking) == k)
        return denoised_ranking

    def topk_related_entities_unsupervised_denoising(self, query_entity, k=10, hth=0.85, c=4):
        # TODO: add logging here, as it's impossible to tell if there was any denoising and how it happened
        # FIXME: also, hth should be got from word_hubness, and only optionally set here, otherwise this will break for
        # different models
        v = self.row_vector_for(query_entity)
        res = self.topk_related_entities(v, k=k)
        # FILTER BAD
        # filter bad ones based on hubness
        filtered_res = []
        filtered_out_root_entities = []
        for e, s in res:
            if self.word_hubness[e] < hth:
                filtered_res.append((e, s))
            else:
                filtered_out_root_entities.append((e, self.word_hubness[e]))
        #print(filtered_res)
        num_swaps = k - len(filtered_res)
        #print(num_swaps)
        # OBTAIN GOOD REPLACEMENTS
        # obtain vectors
        X = []
        for el, d in filtered_res:
            v = self.M_R.get_vector(el)
            X.append(v)
        X = np.asarray(X)
        if len(X) > c:
            num_clusters = c  # as specified in input parameter
        else:
            num_clusters = len(X) - 1  # to avoid error and still get something out of this
        if len(X) == 0:
            return res  # conservative here -- probably want to change
        try:
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans = kmeans.fit(X)
            labels = kmeans.predict(X)
            centroids = kmeans.cluster_centers_
        except OverflowError:
            return res  # this won't be common, so just fallback safely
        except ValueError:
            print(str(X))
            print(str(query_entity))
            print(str(type(query_entity)))
            print(str(v))
            print(str(filtered_res))
        clusters = defaultdict(list)
        for i, entry in enumerate(zip(filtered_res, labels)):
            ranking_entry, label = entry
            clusters[label].append(ranking_entry)
        # Voting session on clusters
        cluster_votes = defaultdict(lambda: defaultdict(int))
        for cid, entities in clusters.items():
            for entity, d in entities:
                v = self.row_vector_for(entity)
                res = self.topk_related_entities(v, k=k)
                for e, d in res:
                    cluster_votes[cid][e] += 1
        # FIXME: remove those in root ranking
        # Big heuristic
        density_votes_cluster = dict()
        for cid, mv in cluster_votes.items():
            total_entities = len(mv)
            total_count = sum(mv.values())
            density = float(float(total_count) / float(total_entities))
            density_votes_cluster[cid] = density
        chosen_cid = None
        max_dens = -1
        for key, v in density_votes_cluster.items():
            if v > max_dens:
                max_dens = v
                chosen_cid = key
        # retrieve votes after filtering hub-bad entities and filtering out entities in root ranking
        root_entities = {e for e, _ in filtered_res}
        root_entities.add(query_entity)
        filtered_cluster_votes = []
        for e, count in cluster_votes[chosen_cid].items():
            if e not in root_entities:
                filtered_cluster_votes.append((e, count))
        filtered_cluster_votes = sorted(filtered_cluster_votes, key=lambda x: x[1], reverse=True)
        filtered_cluster_votes_hub_filtered = []
        for e, count in filtered_cluster_votes:
            if self.word_hubness[e] < hth:
                filtered_cluster_votes_hub_filtered.append((e, count))
        #print(filtered_cluster_votes_hub_filtered)
        final_ranking = filtered_res + filtered_cluster_votes_hub_filtered[:num_swaps]
        #print(len(final_ranking))
        if len(final_ranking) < k:
            filtered_out_root_entities = sorted(filtered_out_root_entities, key=lambda x: x[1])
            #print(filtered_out_root_entities[:(k - len(final_ranking))])
            final_ranking = final_ranking + filtered_out_root_entities[:(k - len(final_ranking))]  # complement with fo
        return final_ranking

    def top_relevant_relations(self, vec_e, k=None, simf=SIMF.COSINE):
        topk = []
        for vec, relation in self.relation_iterator_c():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                print(relation + " has vector with NaNs")
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                similarity = np.dot(vec_e, vec)
                #similarity = 1 - cosine(vec_e, vec)
            elif simf == SIMF.EUCLIDEAN:
                similarity = 1 - euclidean(vec_e, vec)
            topk.append((relation, similarity))
        topk = sorted(topk, key=lambda x: x[1], reverse=True)
        if k:
            return topk[:k]
        else:
            return topk

    def topk_relevant_columns(self, vec_e, k=None, simf=SIMF.COSINE):
        topk = []
        for vec, relation, column in self.column_iterator_c():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                similarity = np.dot(vec_e, vec)
                # similarity = self.re_range_score(similarity)
                # similarity = 1 - cosine(vec_e, vec)
            elif simf == SIMF.EUCLIDEAN:
                similarity = 1 - euclidean(vec_e, vec)
            topk.append((column, relation, similarity))
        topk = sorted(topk, key=lambda x: x[2], reverse=True)
        if k:
            return topk[:k]
        else:
            return topk

    def topk_relevant_rows(self, vec_e, k=5, simf=SIMF.COSINE):
        # obtain topk for each relation first
        partial_topks = []
        for relation, _ in self.RE_R.items():
            rel_rows = np.asarray(list(self.RE_R[relation]["rows"].values()))
            sims = np.dot(rel_rows, vec_e.T)

            sims_nan = np.isnan(sims)
            sims_masked = np.ma.masked_array(sims, mask=sims_nan)
            indexes = np.argsort(sims_masked)[::-1]
            valid_indexes = []
            valid_metrics = []
            for idx in indexes:
                if len(valid_indexes) > k:
                    break
                if sims_masked[idx] is ma.masked:
                    continue
                valid_indexes.append(idx)
                valid_metrics.append(sims_masked[idx])

            for idx, metric in zip(valid_indexes, valid_metrics):
                t = (relation, idx, metric)
                partial_topks.append(t)
        # now get topk of the topks
        topks = sorted(partial_topks, key=lambda x: x[2], reverse=True)[:k]

        to_return = []
        for relation, idx, sim in topks:
            row = self.resolve_row_idx(idx, relation)
            to_return.append((row, relation, sim))
        return to_return

    def topk_relevant_rows_diverse(self, vec_e, k=10, simf=SIMF.COSINE, div_factor=2):
        # obtain topk for each relation first
        partial_topks = []
        for relation, _ in self.RE_R.items():
            rel_rows = np.asarray(list(self.RE_R[relation]["rows"].values()))
            sims = np.dot(rel_rows, vec_e.T)

            sims_nan = np.isnan(sims)
            sims_masked = np.ma.masked_array(sims, mask=sims_nan)
            indexes = np.argsort(sims_masked)[::-1]
            valid_indexes = []
            valid_metrics = []
            for idx in indexes:
                if len(valid_indexes) > div_factor:
                    break
                if sims_masked[idx] is ma.masked:
                    continue
                valid_indexes.append(idx)
                valid_metrics.append(sims_masked[idx])

            for idx, metric in zip(valid_indexes, valid_metrics):
                t = (relation, idx, metric)
                partial_topks.append(t)
        # now get topk of the topks
        topks = sorted(partial_topks, key=lambda x: x[2], reverse=True)[:k]
        to_return = []
        for relation, idx, sim in topks:
            row = self.resolve_row_idx(idx, relation)
            to_return.append((row, relation, sim))
        return to_return

    def __topk_related_rows(self, vec_e, k=5, simf=SIMF.COSINE):
        # TODO: this implementation is too slow
        # TODO: regardless the impl, we'll need a diversified version of this
        topk = []
        min_el = -1000
        for vec, relation, row_idx in self.row_iterator_r():
            if np.isnan(vec).any():
                # FIXME: we could push this checks to building time, avoiding having bad vectors in the relemb
                continue
            similarity = 0
            if simf == SIMF.COSINE:
                similarity = np.dot(vec_e, vec)
                # similarity = self.re_range_score(similarity)
                # similarity = 1 - cosine(vec_e, vec)
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

    def topk_related_rows_in_relation(self, vec_e, relation, k=5):
        # TODO: what if I want to scope the relations of interest
        return

    """
    Explanation API
    """

    def entity_evidence_related_tables(self, table1, table2):
        # TODO: NOT WORKING WELL RIGHT NOW - may need to be composed from the column-evidence, etc
        # FIXME: give a parameter to 50, or otherwise this is broken
        """
        Given two tables as input, find pairs of entities that make the tables related
        :param table1:
        :param table2:
        :return:
        """
        v1 = self.RE_C[table1]['vector']
        v2 = self.RE_C[table2]['vector']

        sims1 = np.dot(self.M_C.vectors, v1.T)
        sims2 = np.dot(self.M_C.vectors, v2.T)
        indexes1 = np.argsort(sims1)[::-1][:50]
        indexes2 = np.argsort(sims2)[::-1][:50]
        ix_indexes = np.intersect1d(indexes1, indexes2)

        metrics1 = sims1[ix_indexes]
        metrics2 = sims2[ix_indexes]
        metrics = np.mean([metrics1, metrics2], axis=0)

        res = self.M_C.generate_response(ix_indexes, metrics).tolist()

        return res

    def column_evidence_related_tables(self, table1, table2, k=10):
        """
        Given two tables as input, find pairs of columns that make the tables related
        :param table1:
        :param table2:
        :return:
        """
        similarities = []
        cs1 = self.RE_C[table1]['columns']
        cs2 = self.RE_C[table2]['columns']
        for c1, v1 in cs1.items():
            for c2, v2 in cs2.items():
                sim = np.dot(v1, v2)
                t = ((c1, c2), sim)
                similarities.append(t)
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def row_evidence_related_tables(self, table1, table2, k=10):
        """
        Given two tables as input, find pairs of rows of either table that make them be related
        :param table1:
        :param table2:
        :return:
        """
        similarities = []
        rs1 = self.RE_R[table1]['rows']
        rs2 = self.RE_R[table2]['rows']
        for idx1, v1 in rs1.items():
            for idx2, v2 in rs2.items():
                sim = np.dot(v1, v2)
                t = ((idx1, idx2), sim)
                similarities.append(t)
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def entity_evidence_related_columns(self, col1, col2, k=10):
        """
        Given two columns as input, find pairs of entities that make the columns related
        :param col1:
        :param col2:
        :return:
        """
        sims1 = np.dot(self.M_C.vectors, col1.T)
        sims2 = np.dot(self.M_C.vectors, col2.T)
        indexes1 = np.argsort(sims1)[::-1][:(k * 5)]
        indexes2 = np.argsort(sims2)[::-1][:(k * 5)]
        ix_indexes = np.intersect1d(indexes1, indexes2)

        metrics1 = sims1[ix_indexes]
        metrics2 = sims2[ix_indexes]
        metrics = np.mean([metrics1, metrics2], axis=0)

        res = self.M_C.generate_response(ix_indexes, metrics).tolist()

        return res

    """
    Summarization API
    """

    def select_diverse_sample(self, vectors, k=5):
        """
        Given a list of vectors, retrieve K that maximize some diversification score
        :param vectors: list of vectors to summarize
        :param k: the total number of vectors to return. size of the summary
        :return: the indexes of the selected vectors
        """
        assert len(vectors) > k

        seed_index = 0
        seed = vectors[seed_index]
        k_result = []
        k_result.append(seed_index)
        k -= 1
        while k > 0:
            sims = np.dot(vectors, seed.T)
            indexes_sorted_by_sims = np.argsort(sims)[::-1]
            most_dissimilar_index = indexes_sorted_by_sims[-1]
            most_dissimilar_metric = sims[most_dissimilar_index]
            k_result.append(most_dissimilar_index)
            k -= 1
            seed = self.combine([vectors[most_dissimilar_index], seed])  # we keep seed always a vector
        # TODO: along with each selected index, show how many other rows are wi thin X distance from it in this table
        # TODO: more like this - given one tuple, find others in the table similar to it
        # TODO: optimal summary - pick enough tuples so all others are within x distance from the summary
        return k_result

    def db_in_relations_summary(self, k=10):
        """
        Retrieve a diverse sample of size k of type relations from the entire database
        :param k:
        :return:
        """
        id_relation = dict()
        vecs = []
        for idx, obj in enumerate(self.RE_C.items()):
            relation, v = obj
            id_relation[idx] = relation
            vecs.append(v['vector'])
        vecs = np.asarray(vecs)
        kmeans = KMeans(n_clusters=int(k/2))
        kmeans = kmeans.fit(vecs)
        labels = kmeans.predict(vecs)
        clusters = defaultdict(list)
        for idx, el in enumerate(labels):
            clusters[el].append(idx)
        # now pick any random idx from each cluster
        selected_tables = []
        table_idxs = [v[0] for _, v in clusters.items()]
        table_idxs.extend([v[-1] for _, v in clusters.items()])
        for i in table_idxs:
            selected_tables.append(id_relation[i])
        return selected_tables

    def relation_in_rows_summary(self, relation, k=10):
        """
        Retrieve a diverse sample of size k of type rows from the input relation
        :param k:
        :return:
        """
        relation_vecs = np.asarray(list(self.RE_R[relation]['rows'].values()))
        summ = self.select_diverse_sample(relation_vecs, k=k)
        df = pd.read_csv(self.path_to_relations + relation, encoding='latin1')
        rows = []
        for index in summ:
            row = df.iloc[index]
            rows.append(row)
        return rows

    """
    Visualization API
    """

    def visualize_vectors(self, vectors, labels, dim=2):
        # TODO: lack of ability to give labels, this is useless without labels
        """
        Given a list of vectors of dimension n, reduce dimensionality to dim and then plot on figure
        :param vectors:
        :param dim:
        :return:
        """
        assert len(vectors) == len(labels)

        return

    """
    Iterator API
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
    similarity and relatedness between 2 entities API
    """

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
        top_similar = self.more_entities_like(instance, k=k)
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
