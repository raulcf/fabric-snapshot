import spacy


wh_words = ["what", "what for", "when", "where", "which", "who", "whom", "whose", "why", "why don't", "how",
                    "how far", "how long", "how many", "how much", "how old", "why do not"]
punctuation = [",", ".", " ", "?", "!", "-", "_", ":", ";"]
nlp = spacy.load("en_core_web_sm")


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


def encode_input(question, answer, sentence_answer, srl_model):

    # Process question
    q_srl = srl_model.predict_json({"sentence": question})
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
    sa_srl = srl_model.predict_json({"sentence": sentence_answer})
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

    # Process answer
    # find indexes of answer in sa
    index_span = None
    answer_annotations = []
    n_tokens_answer = len(answer.split(' '))
    for w_idx, t in enumerate(sa_tokens):
        window_text = sa_tokens[w_idx:(w_idx + n_tokens_answer - 1)]
        if answer == " ".join([el.text for el in window_text]):
            for t in window_text:
                t_ent = t.ent_type_
                if t_ent != "" and t_ent != " ":
                    answer_annotations.append(t_ent)
            index_span = [w_idx + i for i in range(w_idx + n_tokens_answer)]
            break
    answer_roles = []
    for name, indices in reversed(flatten_sa_srl):
        if len(set(index_span).intersection(indices)) != 0:
            answer_roles.append(name,)
    answer_sequence = answer_roles + answer_annotations

    return question_sequence + answer_sequence, sa_sequence


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

    q_seq = annotate_question(flatten_q_srl_ann_label, "The current NFL champions are the Denver Broncos, who "
                                                       "defeated the Carolina Panthers 24–10 in Super Bowl 50.")

    print(q_seq)


