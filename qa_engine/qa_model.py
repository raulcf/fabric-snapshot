from allennlp.service.predictors import Predictor
from allennlp.models import archival


# Keep loaded model
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

    passage = 'A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.'
    q1 = 'How many partially reusable launch systems were developed?'

    input_json = {'passage': passage, 'question': q1}

    res = predictor.predict_json(input_json)

    print(str(res['best_span_str']))

    # test(path)

    exit()

    a = qa("There is an umbrella waiting for you at the corner of the sky. Not many people", "What is waiting for you?")
    print(a)
