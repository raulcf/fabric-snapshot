import json
import spacy
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")


if __name__ == "__main__":

    input_data_path = "/Users/ra-mit/development/fabric/dev-v1.1.json"
    output_data_path = "/Users/ra-mit/development/fabric/dev-v1.1-edit.json"

    total = 0
    total_broken = 0
    gt_questions = 0
    edit_questions = 0
    no_choices = 0

    with open(input_data_path) as f:
        gt = json.load(f)
    training_data = []  # list of <q, a, l>
    pos_labels = 0
    neg_labels = 0
    dataset = gt['data']
    for article in tqdm(dataset):
        for paragraph in article['paragraphs']:
            paragraph_text = paragraph['context']
            paragraph_sentences = sent_tokenize(paragraph_text)
            sentence_offset = []
            soffset = 0
            for s in paragraph_sentences:
                slen = len(s)
                e = (s, soffset + slen)
                soffset += slen
                sentence_offset.append(e)
            for qa in paragraph['qas']:
                question = qa['question']
                gt_questions += 1
                indices = defaultdict(int)

                for a in qa['answers']:
                    indices[a['answer_start']] += 1
                # only if there is more than one possible index
                if len(indices.keys()) > 1:

                    no_choice = True
                    for a in qa['answers']:
                        a_text = a['text']
                        a_analyzed = nlp(a_text)
                        if len(a_analyzed) > 1:
                            no_choice = False
                        elif len(a_analyzed) == 1 and len(a_analyzed[0]) >= 4:
                            no_choice = False
                        elif not a_analyzed[0].is_digit:
                            no_choice = False
                    if no_choice:
                        print("Q: " + str(question))
                        print("A: " + str(qa['answers']))
                        no_choices += 1

    print("total: " + str(total))
    print("total-br: " + str(total_broken))
    print("no choices: " + str(no_choices))
