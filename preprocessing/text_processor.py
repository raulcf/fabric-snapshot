import re


def tokenize(tuple_str, separator):
    clean_tokens = []
    tokens = tuple_str.split(separator)
    for t in tokens:
        if len(t) < 3:
            continue
        if re.search('[0-9]', t) is not None:
            continue
        t = t.replace('_', ' ')
        t = t.replace('-', ' ')
        t = t.lower()
        t_tokens = t.split(' ')
        for token in t_tokens:
            clean_tokens.append(token)
    return clean_tokens


if __name__ == "__main__":
    print("Text processor")