from qa_engine import qa_model as qa

INDEX_MODE = 'doc'

if INDEX_MODE == 'doc':
    from qa_engine import fst_indexer_doc as fst
elif INDEX_MODE == 'chunk':
    from qa_engine import fst_indexer_chunk as fst


def find_answers(question, extract_fragments=False):
    # First search among the documents
    res = fst.search(question, extract_fragments=extract_fragments)
    answers = []
    print("#candidate documents: " + str(len(res)))
    for hit in res:
        body_content = hit['_source']['body']
        print("")
        print("Score: " + str(hit['_score']))
        print("")
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
    answers = find_answers(q, extract_fragments=True)
    print(q)
    for a in answers:
        print(a)
    q = "Does anyone know whether there will be any specific mgmt pack for VxRail and " \
        "VxRack SDDC once it is fully integrated with Dell gear?"
    print("")
    print("")
    print("")
    answers = find_answers(q, extract_fragments=True)
    print(q)
    for a in answers:
        print(a)

    q = "Do we mix Appliance and Widows for PSC?"
    answers = find_answers(q, extract_fragments=True)
    print(q)
    for a in answers:
        print(a)


