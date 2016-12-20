'''
Generates the data for the annoying brother model

Data is sequences of 5 words. each in integer format

 "Answers" will simply be a matrix operation on questions (i.e. word + 2)
'''
import numpy as np

n = 1000
seq_length = 5
vocab_size = 500

data = np.random.randint(0, high=vocab_size, size=(n, seq_length))
data = np.ndarray.flatten(data)
np.savetxt("ab_data.csv", data, delimiter=",")