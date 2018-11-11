import spacy
import config
import pickle
import numpy as np
from keras.preprocessing import sequence

from qa_engine.passage_selector import deep_metric as DM


wh_words = ["what", "what for", "when", "where", "which", "who", "whom", "whose", "why", "why don't", "how",
                    "how far", "how long", "how many", "how much", "how old", "why do not"]
punctuation = [",", ".", " ", "?", "!", "-", "_", ":", ";"]
nlp = spacy.load("en_core_web_sm")


class AnswerVerifier:

    def __init__(self, av_model_path, path_to_vocab, path_to_maxlen):
        archive = archival.load_archive(config.path_to_srl_model)
        p = Predictor.from_archive(archive, 'semantic-role-labeling')
        self.srl_model = p
        self.av_model = DM.load_model_from_path(av_model_path)
        self.vocab = pickle.load(path_to_vocab)
        self.maxlen = pickle.load(path_to_maxlen)

    def vectorize_sequence(self, sequence):
        vec = [self.vocab[tok] for tok in sequence]
        return vec

    # def verify(self, question, answer, sentence_answer, threshold=0.9):
    #     q_srl = self.srl_model.predict_json({"sentence": question})
    #     sa_srl = self.srl_model.predict_json({"sentence": sentence_answer})
    #
    #     question_answer_sequence, sa_sequence, a_seq = encode_q_sa_a(question, answer, sentence_answer,
    #                                                                      q_srl, sa_srl)
    #     # use this encoding to predict
    #     distance = self.av_model.predict(x=[question_answer_sequence, sa_sequence], batch_size=1, verbose=1)
    #     if distance < threshold:
    #         return True
    #     else:
    #         return False

    def verify(self, xq, xa, threshold=0.9):
        answers = []
        distances = self.av_model.predict(x=[xq, xa], batch_size=16, verbose=1)
        for i in range(len(xq)):
            if distances[i] < threshold:
                answers.append(True)
            else:
                answers.append(False)
        return answers

    def encode_batch(self, batch_q, batch_sa, batch_other_params):
        encoded_batch = []
        batch_result_q_srl = self.srl_model.predict_batch_json(batch_q)
        batch_result_sa_srl = self.srl_model.predict_batch_json(batch_sa)
        for q_srl, sa_srl, params in zip(batch_result_q_srl, batch_result_sa_srl, batch_other_params):
            question, sentence_answer, answer = batch_other_params
            question_answer_sequence, sa_sequence, a_seq = encode_q_sa_a(question, answer, sentence_answer,
                                                                             q_srl, sa_srl)
            # call vectorize before returning
            encoded_batch.append((question_answer_sequence, sa_sequence))

        q_vecs = []
        sa_vecs = []

        for q, sa in encoded_batch:
            q_vec = [self.vocab[tok] for tok in q]
            sa_vec = [self.vocab[tok] for tok in sa]
            q_vecs.append(q_vec)
            sa_vecs.append(sa_vec)
        q_vecs = np.asarray(q_vecs)
        sa_vecs = np.asarray(sa_vecs)
        xq = sequence.pad_sequences(q_vecs, maxlen=self.maxlen, dtype='int32', value=0)
        xa = sequence.pad_sequences(sa_vecs, maxlen=self.maxlen, dtype='int32', value=0)

        return xq, xa


class TreeNode:

    def __init__(self, name):
        self.name = name
        self.indices = set()
        self.children = []

    def add_index(self, index):
        self.indices.add(index)

    def add_node(self, list_names, index):
        self.add_index(index)
        # is leaf?
        if len(list_names) == 0:
            return  # reached leaf
        # otherwise grab first node
        first_node = list_names[0]
        # if node is empty we just return
        if first_node == 'O':
            return
        # else we find child
        found = False
        for ch in self.children:
            if ch.name == first_node:
                ch.add_node(list_names[1:], index)
                found = True
        if not found:
            self.children.append(TreeNode(list_names[0]))  # me
        for ch in self.children:
            if ch.name == first_node:
                ch.add_node(list_names[1:], index)

    def enumerate(self):
        children_enumeration = [ch.enumerate() for ch in self.children]
        flat = []
        for ce in children_enumeration:
            flat += ce
        return [(self.name, sorted(list(self.indices)))] + flat


class SRLTree:

    def __init__(self):
        self.root = TreeNode('root')

    def add_token(self, list_names, index):
        self.root.add_node(list_names, index)

    def enumerate_order(self):
        list_roles_with_indices = self.root.enumerate()
        return list_roles_with_indices


def interpret_srl_result(res):
    tag_tree = []
    words_in_sentence = res['words']
    for w in words_in_sentence:
        tag_tree.append([w])

    tags_col = [verb['tags'] for verb in res['verbs']]
    tag_level = 0
    for tag_col in tags_col:
        for idx, tag in zip(range(len(words_in_sentence)), tag_col):
            formatted_tag = tag[2:] + "_" + str(tag_level) if tag != 'O' else 'O'
            tag_tree[idx].append(formatted_tag)
        tag_level += 1
    return tag_tree


def annotate_question(flatten_srl, question):
    question_sequence = []
    q_tokens = nlp(question)
    for name, indices, needs_annotation in flatten_srl:
        annotations = set()
        if name[0] == "V":
            verbs = [q_tokens[idx] for idx in indices]
            for v in verbs:
                if v.text.strip() != "" and v.text.strip() != " ":
                    annotations.add(v.lemma_)
        elif needs_annotation:
            text_slice = q_tokens[min(indices):max(indices)]
            # entities
            for t in text_slice:
                t_ent = t.ent_type_
                if t_ent != "" and t_ent != " ":
                    annotations.add(t_ent)
            # wh words
            for t in text_slice:
                lower_token = t.text.lower()
                if lower_token in wh_words:
                    annotations.add(lower_token)
        question_sequence.append(name[:-2])  # remove the level metadata which was previously added
        question_sequence.extend(list(annotations))
    return question_sequence


def annotate_sa(flatten_srl, sentence):
    sa_sequence = []
    tokens = nlp(sentence)
    for name, indices, needs_annotation in flatten_srl:
        annotations = set()
        if name[0] == "V":
            verbs = [tokens[idx] for idx in indices]
            for v in verbs:
                if v.text.strip() != "" and v.text.strip() != " ":
                    annotations.add(v.lemma_)
        elif needs_annotation:
            text_slice = tokens[min(indices):max(indices)]
            # entities
            for t in text_slice:
                t_ent = t.ent_type_
                if t_ent != "" and t_ent != " ":
                    annotations.add(t_ent)
        sa_sequence.append(name[:-2])  # remove the level metadata which was previously added
        sa_sequence.extend(list(annotations))
    return sa_sequence, tokens


def flatten_srl_repr(list_srl):
    flatten_tree = SRLTree()
    for idx, el in enumerate(list_srl):
        flatten_tree.add_token(el[1:], idx)
    flatten_srl = flatten_tree.enumerate_order()
    return flatten_srl


def encode_q_sa(question, sentence_answer, q_srl, sa_srl):
    # Process question
    # q_srl = srl_model.predict_json({"sentence": question})
    list_q_srl = interpret_srl_result(q_srl)
    flatten_q_srl = flatten_srl_repr(list_q_srl)
    flatten_q_srl_ann_label = []
    seen = set()
    for name, indices in reversed(flatten_q_srl):
        if len(set(indices).intersection(seen)) == 0:
            flatten_q_srl_ann_label.append((name, indices, True))
        else:
            flatten_q_srl_ann_label.append((name, indices, False))
        seen.update(set(indices))
    flatten_q_srl_ann_label = list(reversed(flatten_q_srl_ann_label))
    flatten_q_srl_ann_label = flatten_q_srl_ann_label[1:]  # remove root node of tree
    question_sequence = annotate_question(flatten_q_srl_ann_label, question)

    # Process sentence-answer
    # sa_srl = srl_model.predict_json({"sentence": sentence_answer})
    list_sa_srl = interpret_srl_result(sa_srl)
    flatten_sa_srl = flatten_srl_repr(list_sa_srl)
    flatten_sa_srl_ann_label = []
    seen = set()
    for name, indices in reversed(flatten_sa_srl):
        if len(set(indices).intersection(seen)) == 0:
            flatten_sa_srl_ann_label.append((name, indices, True))
        else:
            flatten_sa_srl_ann_label.append((name, indices, False))
        seen.update(set(indices))
    flatten_sa_srl_ann_label = list(reversed(flatten_sa_srl_ann_label))
    flatten_sa_srl_ann_label = flatten_sa_srl_ann_label[1:]  # remove root node of tree
    sa_sequence, sa_tokens = annotate_sa(flatten_sa_srl_ann_label, sentence_answer)

    return question_sequence, sa_sequence, sa_tokens, flatten_sa_srl


def encode_q_sa_a(question, answer, sentence_answer, q_srl, sa_srl):

    question_sequence, sa_sequence, sa_tokens, flatten_sa_srl = encode_q_sa(question, sentence_answer, q_srl, sa_srl)

    a_tokens = nlp(answer)
    answer_annotations = set()

    window_size = len(a_tokens)
    index_span = None
    for i in range(len(sa_tokens)):
        span = sa_tokens[i: i + window_size]
        if " ".join([e.text for e in a_tokens]) == " ".join([e.text for e in span]):
            for t in span:
                t_ent = t.ent_type_
                if t_ent != "" and t_ent != " ":
                    answer_annotations.add(t_ent)
            index_span = [i + j for j in range(window_size)]
            break

    answer_roles = []
    for name, indices in reversed(flatten_sa_srl):
        if name == 'root':
            continue  # for control, but does not provide any info

        if index_span is None:
            print("QUESTION: " + str(question))
            print("ANSWER: " + str(answer))
            print("SA: " + str(sentence_answer))
            print("SPAN: " + str(span))

        if index_span is None:
            return None, None, None

        if len(set(index_span).intersection(indices)) != 0:
            answer_roles.append(name)
    answer_sequence = answer_roles + list(answer_annotations)

    return question_sequence + answer_sequence, sa_sequence, answer_sequence


    # # Process answer
    # # find indexes of answer in sa
    # index_span = None
    # answer_annotations = set()
    # found_commas = answer.count(',')
    # found_apost = answer.count("'s")
    # found_quotes = answer.count('\"')
    # n_tokens_answer = len(answer.split(' ')) + found_commas + found_apost + found_quotes
    # for w_idx, t in enumerate(sa_tokens):
    #     end_span = w_idx + n_tokens_answer
    #     window_text = sa_tokens[w_idx:end_span]
    #     # now we need to remove commas so the joining works ok here
    #     answer = answer.replace(',', '')
    #     answer = answer.replace("'s", '')
    #     answer = answer.replace("\"", '')
    #     if answer == " ".join([el.text for el in window_text if el.text != "," and el.text != "'s" and el.text != '\"']):
    #         for t in window_text:
    #             t_ent = t.ent_type_
    #             if t_ent != "" and t_ent != " ":
    #                 answer_annotations.add(t_ent)
    #         index_span = [w_idx + i for i in range(n_tokens_answer)]
    #         break
    # answer_roles = []
    # for name, indices in reversed(flatten_sa_srl):
    #     if name == 'root':
    #         continue  # for control, but does not provide any info
    #
    #     if index_span is None:
    #         print(answer)
    #         print(sentence_answer)
    #         print(window_text)
    #
    #     if len(set(index_span).intersection(indices)) != 0:
    #         answer_roles.append(name)
    # answer_sequence = answer_roles + list(answer_annotations)
    #
    # return question_sequence + answer_sequence, sa_sequence, answer_sequence


def _encode_q_sa_a(question, answer, sentence_answer, q_srl, sa_srl):

    question_sequence, sa_sequence, sa_tokens, flatten_sa_srl = encode_q_sa(question, sentence_answer, q_srl, sa_srl)

    # Process answer
    # find indexes of answer in sa
    index_span = None
    answer_annotations = set()
    found_commas = answer.count(',')
    found_apost = answer.count("'s")
    found_quotes = answer.count('\"')
    n_tokens_answer = len(answer.split(' ')) + found_commas + found_apost + found_quotes
    for w_idx, t in enumerate(sa_tokens):
        end_span = w_idx + n_tokens_answer
        window_text = sa_tokens[w_idx:end_span]
        # now we need to remove commas so the joining works ok here
        answer = answer.replace(',', '')
        answer = answer.replace("'s", '')
        answer = answer.replace("\"", '')
        if answer == " ".join([el.text for el in window_text if el.text != "," and el.text != "'s" and el.text != '\"']):
            for t in window_text:
                t_ent = t.ent_type_
                if t_ent != "" and t_ent != " ":
                    answer_annotations.add(t_ent)
            index_span = [w_idx + i for i in range(n_tokens_answer)]
            break
    answer_roles = []
    for name, indices in reversed(flatten_sa_srl):
        if name == 'root':
            continue  # for control, but does not provide any info

        if index_span is None:
            print(answer)
            print(sentence_answer)
            print(window_text)

        if len(set(index_span).intersection(indices)) != 0:
            answer_roles.append(name)
    answer_sequence = answer_roles + list(answer_annotations)

    return question_sequence + answer_sequence, sa_sequence, answer_sequence


if __name__ == "__main__":
    print("Answer verifier API")

    from allennlp.service.predictors import Predictor
    from allennlp.models import archival
    import config

    archive = archival.load_archive(config.path_to_srl_model)
    psrl = Predictor.from_archive(archive, 'semantic-role-labeling')

    res = psrl.predict_json({"sentence": "Which NFL team represented the AFC at Super Bowl 50?"})
    res2 = psrl.predict_json({"sentence": "The current NFL champions are the Denver Broncos, who\
     defeated the Carolina Panthers 24–10 in Super Bowl 50."})
    res3 = psrl.predict_json({"sentence": "i run fast when the field who was singing entered the room."})

    res_i = interpret_srl_result(res2)

    tree = SRLTree()
    for idx, t in enumerate(res_i):
        # if t[0] not in punctuation:
        tree.add_token(t[1:], idx)

    serial_order = tree.enumerate_order()

    flatten_q_srl_ann_label = []
    seen = set()
    for name, indices in reversed(serial_order):
        if len(set(indices).intersection(seen)) == 0:
            flatten_q_srl_ann_label.append((name, indices, True))
        else:
            flatten_q_srl_ann_label.append((name, indices, False))
        seen.update(set(indices))
    flatten_q_srl_ann_label = list(reversed(flatten_q_srl_ann_label))
    flatten_q_srl_ann_label = flatten_q_srl_ann_label[1:]  # remove root node of tree

    question = "Which NFL team represented the AFC at Super Bowl 50?"
    sa = "The current NFL champions are the Denver Broncos, who defeated the Carolina Panthers 24–10 in Super Bowl 50."
    answer = "Denver Broncos"

    question_answer_sequence, sa_sequence = encode_q_sa_a(question, answer, sa, psrl)

    print("Q-A--")
    print(question_answer_sequence)
    print("SA--")
    print(sa_sequence)
    print("--")

    # q_seq = annotate_question(flatten_q_srl_ann_label, "The current NFL champions are the Denver Broncos, who "
    #                                                    "defeated the Carolina Panthers 24–10 in Super Bowl 50.")

    # print(q_seq)


