import numpy as np

n = 1000
Q = np.genfromtxt('ab_data.csv',delimiter=',')
Q = np.reshape(Q,(n,5))
index = 0
batch_size = 2



def RealAnswer(q_in):
    #Contains Logic to make the real answer from the question
    #for now, just adding 2 to every word (annoying right?)
    #a one hot answer!
    key = 1*np.ones_like(q_in)
    real_answer = q_in + key
    return convert_onehot(real_answer,10)

def convert_onehot(a,classes):
    z = (np.arange(classes) == a[:,:,None]-1).astype(int)
    #z = np.zeros(list(a.shape) + [classes])
    #print(z)
    #z[list(np.indices(z.shape[:-1])) + [a]] = 1
    return z
'''
print("q",q)
qd = np.reshape(q,(5,1))
print("qd",qd)
answer = RealAnswer(qd)
print("a",answer)

xs = np.concatenate((qd,answer),axis=0)
print("combo",xs)
'''

qg = Q[index:index+batch_size]
print("qg",qg)
index += batch_size

#grab real answer from question, resized for discriminator (batch,seq_length,features)
qd = np.reshape(qg,(batch_size,5,1))
print("qd",qd)
answer = RealAnswer(qd)
print("a",answer)
xs = np.concatenate((qd,answer),axis=1)
print("combo",xs)