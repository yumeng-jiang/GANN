import pandas as pd
import numpy as np

number_set = ('zero','one','two','three','four','five','six','seven','eight','nine')
with open("glove/glove.6B.100d.txt", "r") as ins:
    matrix = []
    for line in ins:
        if line.split(None, 1)[0] in number_set:
            matrix.append(line.split())
    print(matrix)
    
embedding = []
embedding.append(matrix[9][1:])
for _ in range(9):
    embedding.append(matrix[_][1:])
embedding
float_embedding = np.array(embedding).astype('float')
np.savetxt('word2vec.csv',float_embedding)