from allennlp.common.util import JsonDict
from allennlp.service.predictors import BidafPredictor


def qa(passage, question):
    input_json = {'passage': passage, 'question': question}
    # payload = JsonDict.fromkeys()

    bp = BidafPredictor()
    res = bp.predict_json(input_json)

    return res


if __name__ == "__main__":
    print("QA ")

    a = qa("There is an umbrella waiting for you at the corner of the sky", "What is waiting for you?")
    print(a)
