#load data from Cifar100 file
def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def load_cifar():

    #we could randomly pick coarse label  or fine label
    #for now just return fine label

    data_file = "cifar/train"
    meta_file = "cifar/meta"
    dic = unpickle(meta_file)
    fine_label_names = dic['fine_label_names'] #list of 100 label names
    #coarse_label_names = dic['coarse_label_names']

    dic = unpickle(data_file)
    data = dic['data']
    data_fine_labels = dic['fine_labels']

    return (data, data_fine_labels, fine_label_names)

# load cifar 100 file with word2vec embeddings
def load_cifar_word2vec():
    import gensim

    # loading cifar data
    cifar_data = load_cifar()
    # loading word2vec model
    model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # need to change "aquarium_fish" to "fish" and "sweet_pepper" to "bell_pepper" so that the words appear in the word2vec model
    cifar_data[2][1] = 'fish'
    cifar_data[2][83] = 'bell_pepper'

    # creating the vector embeddings for each label
    vector_emb = model[cifar_data[2]]

    return(cifar_data[0],cifar_data[1],vector_emb)
