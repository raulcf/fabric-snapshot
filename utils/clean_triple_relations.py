from dataaccess import csv_access
from preprocessing import text_processor as tp
import pandas as pd


def lines_in_files(path):
    all_files = csv_access.list_files_in_directory(path)
    total = 0
    for f in all_files:
        with open(f, "r") as g:
            lines = g.readlines()
            total += len(lines)
    print("Total lines: " + str(total))


def statistics_words(path):
    all_files = csv_access.list_files_in_directory(path)
    total_tokens = 0
    total_lines = 0
    for f in all_files:
        with open(f, "r") as g:
            lines = g.readlines()
            total_lines += len(lines)
            for l in lines:
                tokens = l.split(" ")
                total_tokens += len(tokens)
    print("Total tokens: " + str(total_tokens))
    print("Avg tokens: " + str(float(total_tokens/total_lines)))

if __name__ == "__main__":
    print("Process relations")

    # statistics_words("/Users/ra-mit/data/fabric/academic/clean_relations/")
    # exit()

    path = "/Users/ra-mit/data/fabric/academic/triple_relations/"
    out_path = "/Users/ra-mit/data/fabric/academic/clean_triple_relations/"

    pronouns = ["She", "He", "she", "he", "I", "i"]

    all_files = csv_access.list_files_in_directory(path)

    #all_files = [all_files[0]]

    for fpath in all_files:
        name = (fpath.split("/")[-1]).split(".")[0]

        lines = []
        print("Processing: " + str(fpath))
        df = pd.read_csv(fpath, encoding='latin1')
        for index, row in df.iterrows():
            s = row['s']
            p = row['p']
            o = row['o']
            # clean stuff
            s_tokens = tp.tokenize(s, " ", min_token_length=1)
            s_tokens = [el.strip() for el in s_tokens]
            for idx in range(len(s_tokens)):
                if s_tokens[idx] in pronouns:
                    s_tokens[idx] = name
            p_tokens = tp.tokenize(p, " ", min_token_length=1)
            p_tokens = [el.strip() for el in p_tokens]
            for idx in range(len(p_tokens)):
                if p_tokens[idx] in pronouns:
                    p_tokens[idx] = name
            o_tokens = tp.tokenize(o, " ", min_token_length=1)
            o_tokens = [el.strip() for el in o_tokens]
            for idx in range(len(o_tokens)):
                if o_tokens[idx] in pronouns:
                    o_tokens[idx] = name
            clean_s = " ".join(s_tokens)
            clean_o = " ".join(o_tokens)
            clean_p = " ".join(p_tokens)
            if clean_s != "" and clean_p != "" and clean_o != "":
                line = clean_s + "," + clean_p + "," + clean_o
                lines.append(line)
        with open(out_path + name + ".csv", "w") as g:
            g.write("s,p,o\n")
            for el in lines:
                g.write(el + '\n')



        # with open(fpath, "r") as f:
        #     relations = f.readlines()
        #     for r in relations:
        #         tokens = r.split(" ")[1::]  # remove number
        #         pre_tokens = tp.tokenize(" ".join(tokens), " ")  # clean stuff
        #         pre_tokens = [el.strip() for el in pre_tokens]
        #         # change pronouns by names
        #         for idx in range(len(pre_tokens)):
        #             tk = pre_tokens[idx]
        #             if tk in pronouns:
        #                 pre_tokens[idx] = name
        #         # # add name if not present already
        #         # if name not in pre_tokens:
        #         #     pre_tokens.append(name)
        #         if len(pre_tokens) > 0:
        #             pre_processed_tokens.append(set(pre_tokens))
        #
        #     # Remove near-duplicates
        #     idx_to_remove = set()
        #     for i in range(len(pre_processed_tokens)):
        #         for j in range(len(pre_processed_tokens)):
        #             if i == j:
        #                 continue  # do not remove yourself
        #             si = pre_processed_tokens[i]
        #             sj = pre_processed_tokens[j]
        #             js = len(si.intersection(sj)) / len(si.union(sj))
        #             if js > 0.6:
        #                 idx_to_remove.add(j)
        #     for idx in idx_to_remove:
        #         pre_processed_tokens[idx] = set()
        #
        #     clean_tokens_set = [tk for tk in pre_processed_tokens if len(tk) > 2]
        #
        #     clean_rels = []
        #     for s in clean_tokens_set:
        #         rel = ",".join(s)
        #         print(rel)
        #         clean_rels.append(rel)
        #
        #     with open(out_path + name + ".txt", "w") as g:
        #         for el in clean_rels:
        #             g.write(el + '\n')
