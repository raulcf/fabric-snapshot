import nltk

from nltk.corpus import stopwords
import spacy


stop_words = set(stopwords.words('english'))
wh_words = ["what", "what for", "when", "where", "which", "who", "whom", "whose", "why", "why don't", "how",
                    "how far", "how long", "how many", "how much", "how old", "why do not"]
punctuation = [",", ".", " ", "?", "!", "-", "_", ":", ";"]
nlp = spacy.load("en_core_web_lg")


def tokenize_and_filter_stopwords(string):
    tokens = nlp(string)
    tokens = [t for t in tokens if t.text not in stop_words and t.text not in wh_words and t.text not in punctuation]
    return tokens


def select_noun_verb(tokens):
    tokens = [t for t in tokens if t.pos_ in ['NOUN', 'PROPN', 'VERB']]
    return tokens


def select_verb(tokens):
    tokens = [t for t in tokens if t.pos_ == 'VERB']
    return tokens


def compatible_max_oneside_to_other(question, sentence, threshold=0.7):
    # remove stopwords of question and sentence
    q_tokens = tokenize_and_filter_stopwords(question)
    s_tokens = tokenize_and_filter_stopwords(sentence)
    # obtain max sim between every word in one side with respect to all words in other side
    max_sim = []
    for q_t in q_tokens:
        max_q_t_sim = []
        for s_t in s_tokens:
            if q_t.has_vector and s_t.has_vector:
                sim = q_t.similarity(s_t)
                max_q_t_sim.append(sim)
        maximum = max(max_q_t_sim)
        max_sim.append(maximum)
    avg_sim = sum(max_sim) / len(max_sim)
    if avg_sim > threshold:
        return True, avg_sim
    return False, avg_sim


def compatible_only_noun_verb(question, sentence, threshold=0.7):
    # remove stopwords of question and sentence
    q_tokens = tokenize_and_filter_stopwords(question)
    s_tokens = tokenize_and_filter_stopwords(sentence)

    # retain only nouns and verbs
    q_tokens = select_noun_verb(q_tokens)
    s_tokens = select_noun_verb(s_tokens)
    # obtain max sim between every word in one side with respect to all words in other side
    sims = []
    for q_t in q_tokens:
        for s_t in s_tokens:
            if q_t.has_vector and s_t.has_vector:
                if q_t.text != s_t.text:
                    sim = q_t.similarity(s_t)
                    sims.append(sim)
    avg_sim = sum(sims) / len(sims)
    if avg_sim > threshold:
        return True, avg_sim
    return False, avg_sim


def compatible(question, sentence, threshold=0.7):
    # remove stopwords of question and sentence
    q_tokens = tokenize_and_filter_stopwords(question)
    s_tokens = tokenize_and_filter_stopwords(sentence)

    # retain only nouns and verbs
    q_tokens = select_verb(q_tokens)
    s_tokens = select_verb(s_tokens)
    # obtain max sim between every word in one side with respect to all words in other side
    max_sim = []
    for q_t in q_tokens:
        local_sim = []
        for s_t in s_tokens:
            if q_t.has_vector and s_t.has_vector:
                # if q_t.text != s_t.text:
                sim = q_t.similarity(s_t)
                local_sim.append(sim)
        max_local_sim = max(local_sim)
        max_sim.append(max_local_sim)
    avg_sim = sum(max_sim) / len(max_sim)
    if avg_sim > threshold:
        return True, avg_sim
    return False, avg_sim


if __name__ == "__main__":
    print("Main of compatibility api")

    q = "Which NFL team represented the AFC at Super Bowl 50?"
    s = "Along with the Colts, the Cleveland Browns and the Pittsburgh Steelers agreed to join the\
     10 AFL teams to form the AFC."
    print(compatible(q, s))

    s = "The Denver Broncos would later become the second AFC team and fourth club overall to wear\
     white jerseys in a Super Bowl despite being the home team in Super Bowl 50."
    print(compatible(q, s))

    s = "The American Football Conference (AFC) champion Denver Broncos defeated the National Football\
     Conference (NFC) champion Carolina Panthers 24–10 to win their third Super Bowl championship."
    print(compatible(q, s))

    s = "When the NFL realigned for the 2002 schedule, the newly created AFC South was formed from two\
     former AFC Central teams, an AFC East team, and an expansion team."
    print(compatible(q, s))

    s = "The Super Bowl Champions is an annual documentary series created by NFL Films (broadcast on\
     NFL Network and CBS)."
    print(compatible(q, s))

    s = "The Super Bowl XXV logo was painted at midfield, and the NFL 75th Anniversary logo was painted\
     at midfield in Super Bowl XXIX."
    print(compatible(q, s))


