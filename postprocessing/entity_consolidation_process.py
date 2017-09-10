import numpy as np

path_results = '/Users/ra-mit/research/data-discovery/papers/fabric-paper/eval_results/entity_consolidation/results_1name.txt'
path_gt = '/Users/ra-mit/research/data-discovery/papers/fabric-paper/eval_results/entity_consolidation/entity_consolidation_gt.csv'

with open(path_results, "r") as f:
    lines_res = f.readlines()

with open(path_gt, "r") as f:
    lines_gt = f.readlines()

not_found_set = set()

total_found = 0
total_gt = 0
lens = []
for res, gt in zip(lines_res, lines_gt):
    tokens_res = set(res.split(","))
    tokens_res = set([el.strip() for el in tokens_res])
    print(tokens_res)
    tokens_gt = set(gt.split(","))
    tokens_gt = set([el.strip() for el in tokens_gt])
    lens.append(len(tokens_gt))
    total_gt += len(tokens_gt)
    print(tokens_gt)
    print("--")
    e_found = tokens_res.intersection(tokens_gt)
    total_found += len(e_found)

    not_found_set |= tokens_gt - tokens_res

print("entities stats")
lens = np.asarray(lens)
min = np.min(lens)
max = np.max(lens)
avg = np.mean(lens)
median = np.percentile(lens, 50)
print("min: " + str(min))
print("max: " + str(max))
print("avg: " + str(avg))
print("median: " + str(median))

print("1st iter")
print("GT: " + str(total_gt))
print("Found: " + str(total_found))
print("%: " + str(total_found/total_gt))

print("not found: " + str(len(not_found_set)))
print(not_found_set)

second_iter = 7

total_found += 7

print("2nd iter")
print("GT: " + str(total_gt))
print("Found: " + str(total_found))
print("%: " + str(total_found/total_gt))


# concept expansion

path_c_gt = '/Users/ra-mit/research/data-discovery/papers/fabric-paper/eval_results/concept_expansion/concept_expansion_gt.txt'

with open(path_c_gt, "r") as f:
    lines_gt = f.readlines()

for l in lines_gt:
    tokens = l.split(",")
    print("Entity: " + str(tokens[0]))
    print("Num entities: " + str(len(tokens) - 1))

