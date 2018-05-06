from qa_engine import qa_model as qa

from qa_engine import fst_indexer_doc as fst
from qa_engine import fst_indexer_chunk as fsc


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
