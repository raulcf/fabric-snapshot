from nltk.tokenize import sent_tokenize

from qa_engine import qa_model as qa
from qa_engine import fst_indexer_doc as fst
from qa_engine import fst_indexer_chunk as fsc
from qa_engine.passage_selector.answer_predictor_api import AnswerPredictor


def find_answers_docs(question, extract_fragments=False, host=None, limit_results=3):
    # First search among the documents
    res = fst.search(question, extract_fragments=extract_fragments, host=host)
    res = res[:limit_results]
    answers = []
    #print("#candidate documents: " + str(len(res)))
    for hit in res:
        body_content = hit['_source']['body']
        #print("")
        #print("Score: " + str(hit['_score']))
        #print("")
        if extract_fragments:
            passages = hit['highlight']['body']
            for passage in passages:
                answer = qa.qa(passage, question)
                answers.append(answer)
        elif not extract_fragments:
            answer = qa.qa(body_content, question)
            answers.append(answer)
    return answers

global log_info
log_info = dict()
log_info['non_first_passage'] = 0

def get_log_info():
    global log_info
    return log_info

def select_passages_first_nonbad(question, answer_predictor: AnswerPredictor, host=None, k=1):
    """
    Select first passage with at least one sentence candidate
    """
    res = fsc.search(question, host=host)
    passages = [hit['_source']['body'] for hit in res[:k]]
    for idx, passage in enumerate(passages):
        sentences = sent_tokenize(passage)
        for sentence in sentences:
            prediction, distance = answer_predictor.is_answer(question, sentence)
            if prediction:
                if idx > 0:
                    global log_info
                    log_info['non_first_passage'] += 1
                return [passage]
    return [passages[0]]

#def select_passages_shortest_distance(question, answer_predictor: AnswerPredictor, host=None, k=1):
def select_passages(question, answer_predictor: AnswerPredictor, host=None, k=1):
    """
    Select passage that contains the sentence candidate with the shortest distance
    """
    res = fsc.search(question, host=host)
    passages = [hit['_source']['body'] for hit in res[:k]]
    chosen_passage = None
    min_distance = 1000
    for passage in passages:
        sentences = sent_tokenize(passage)
        for sentence in sentences:
            prediction, distance = answer_predictor.is_answer(question, sentence)
            if prediction and distance < min_distance:
                min_distance = distance
                chosen_passage = passage
    if chosen_passage is None:
        #print(".")
        chosen_passage = passages[0]  # fall back down into baseline method
    return [chosen_passage]

def select_passages_more_answers(question, answer_predictor: AnswerPredictor, host=None, k=1):
    """
    Select passage that contains the highest number of sentence candidates
    """
    # Search for chunks and get the first passages up to k
    res = fsc.search(question, host=host)
    passages = [hit['_source']['body'] for hit in res[:k]]
    # Now, let's obtain a score per passage. How many sentences may be an answer/total sentences
    chosen_passage = None
    current_max_score = -1
    for passage in passages:
        sentences = sent_tokenize(passage)
        positive_predictions = 0
        for sentence in sentences:
            prediction, distance = answer_predictor.is_answer(question, sentence)
            if prediction:
                positive_predictions += 1
        score = float(float(positive_predictions) / float(len(sentences)))
        #print("Score passage: " + str(score))
        if score > current_max_score and current_max_score > 0:
            chosen_passage = passage
    if chosen_passage is None:
        #print(".")
        chosen_passage = passages[0]  # fall back down into baseline method
    return [chosen_passage]


def dummy_select_passages(question, host=None, k=1):
    """
    This is the baseline used before having a passage selector
    :param question:
    :param host:
    :param k:
    :return:
    """
    res = fsc.search(question, host=host)
    res = res[:k]
    return [hit['_source']['body'] for hit in res]


def find_answers_chunks(question, extract_fragments=False, host=None, limit_results=3):
    # First search among the documents
    res = fsc.search(question, host=host)
    res = res[:limit_results]
    answers = []
    for hit in res:
        body_content = hit['_source']['body']
        if extract_fragments:
            passages = hit['highlight']['body']
            for passage in passages:
                answer = qa.qa(passage, question)
                answers.append(answer)
        elif not extract_fragments:
            answer = qa.qa(body_content, question)
            answers.append(answer)
    return answers

if __name__ == "__main__":
    print("QA API")

    # basic tests

    q = "When will EHC support Unity?"
    answers = find_answers_docs(q, extract_fragments=True)
    print(q)
    for a in answers:
        print(a)
    q = "Does anyone know whether there will be any specific mgmt pack for VxRail and " \
        "VxRack SDDC once it is fully integrated with Dell gear?"
    print("")
    print("")
    print("")
    answers = find_answers_docs(q, extract_fragments=True)
    print(q)
    for a in answers:
        print(a)

    q = "Do we mix Appliance and Widows for PSC?"
    answers = find_answers_docs(q, extract_fragments=True)
    print(q)
    for a in answers:
        print(a)
