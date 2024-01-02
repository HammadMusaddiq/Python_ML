import numpy as np
from util import write_w2v, writeAnalogies, writeGroupAnalogies, convert_legacy_to_keyvec, load_legacy_w2v, pruneWordVecs

path1 = 'data/w2vs/reddit.US.txt.tok.clean.cleanedforw2v_0.w2v'
path2 = 'output/gender_evalset/sum/gender_sum_biasedEmbeddingsOut.w2v'
word_vectors1, embedding_dim1 = load_legacy_w2v(path1)
word_vectors2, embedding_dim2 = load_legacy_w2v(path2)

print(set(word_vectors1) - set(word_vectors2))

