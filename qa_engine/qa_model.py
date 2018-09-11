from allennlp.service.predictors import Predictor
from allennlp.models import archival
import config


# Keep loaded model
predictor = None
initialized = False
path_to_model = config.path_to_bidaf_model


class QAModel:

    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        archive = archival.load_archive(path_to_model)
        p = Predictor.from_archive(archive, 'machine-comprehension')
        # global predictor
        self.predictor = p

    def qa(self, passage, question):
        # if not initialized:
        #     init(path_to_model)
        input_json = {'passage': passage, 'question': question}
        res = self.predictor.predict_json(input_json)
        return res['best_span_str']


def init(path_to_model):
    archive = archival.load_archive(path_to_model)
    p = Predictor.from_archive(archive, 'machine-comprehension')
    global predictor
    predictor = p


def qa(passage, question):
    if not initialized:
        init(path_to_model)
    input_json = {'passage': passage, 'question': question}
    res = predictor.predict_json(input_json)
    return res['best_span_str']


def qa_raw(passage, question):
    if not initialized:
        init(path_to_model)
    input_json = {'passage': passage, 'question': question}
    res = predictor.predict_json(input_json)
    return res


def qa_batch_raw(batch):
    if not initialized:
        init(path_to_model)
    res = predictor.predict_batch_json(batch)
    return res


def qa_batch(batch):
    if not initialized:
        init(path_to_model)
    res = predictor.predict_batch_json(batch)
    return [a['best_span_str'] for a in res]


if __name__ == "__main__":
    print("QA ")

    # path = "/Users/ra-mit/Downloads/bidaf-model-2017.09.15-charpad.tar.gz"
    #
    # archive = archival.load_archive(path)
    # predictor = Predictor.from_archive(archive, 'machine-comprehension')
    #
    # passage = 'A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable
    # of '\
    #           'launching a payload into space more than once. This contrasts with expendable launch systems, where ' \
    #           'each launch vehicle is launched once and then discarded. No completely reusable orbital launch system ' \
    #           'has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and ' \
    #           'Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle ' \
    #           'main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were ' \
    #           'reused after several months of refitting work for each launch. The external tank was discarded after ' \
    #           'each flight.'
    # q1 = 'How many partially reusable launch systems were developed?'
    #
    # input_json = {'passage': passage, 'question': q1}
    #
    # res = predictor.predict_json(input_json)
    #
    # print(str(res['best_span_str']))
    #
    # # test(path)
    #
    # exit()

    passage = "There is an umbrella waiting for you at the corner of the sky. It will be only when the bodies of the "
    "peoples in the cities are lining to the mountain that the sky will come down. And only then, with the sky"
    "on its knees and the mountains longing the stream of life, the umbrella will take its realm"
    question = "What is waiting for you at the corner of the sky?"

    # passage = "Once upon a time there was a short person named Victor. Victor used to be short, stupid and" \
    #           "had a strange taste for shiny things. In other words, Victor was a dumb. Raul on the other hand, " \
    #           "was not."
    #
    # question = "Who is not a dumb?"

    if not initialized:
        init(path_to_model)
    input_json = {'passage': passage, 'question': question}
    res = predictor.predict_json(input_json)

    print(res)

    # a = qa(passage, question)
    # print("Question: ")
    # print(str(question))
    # print("")
    # print("Answer:")
    # print(str(a))
