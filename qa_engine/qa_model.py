from allennlp.common.util import JsonDict
from allennlp.service.predictors import BidafPredictor, Predictor
from allennlp.models import archival
from allennlp.models import bidaf
from allennlp.models.bidaf import BidirectionalAttentionFlow


predictor = None


def init(path_to_model):
    archive = archival.load_archive(path_to_model)
    p = Predictor.from_archive(archive, 'machine-comprehension')
    global predictor
    predictor = p


def qa(passage, question):
    input_json = {'passage': passage, 'question': question}

    res = predictor.predict_json(input_json)

    return res

if __name__ == "__main__":
    print("QA ")

    path = "/Users/ra-mit/Downloads/bidaf-model-2017.09.15-charpad.tar.gz"

    archive = archival.load_archive(path)
    predictor = Predictor.from_archive(archive, 'machine-comprehension')

    passage = "There is an umbrella waiting for you at the corner of the sky. Although not many people know how" \
              "to read the truth these days, there is still a sense of glory in the air, but glory of a past epoch."
    q1 = "What is waiting for you?"

    input_json = {'passage': passage, 'question': q1}

    res = predictor.predict_json(input_json)

    print(str(res))

    # test(path)

    exit()

    a = qa("There is an umbrella waiting for you at the corner of the sky. Not many people", "What is waiting for you?")
    print(a)
