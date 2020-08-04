import numpy as np

def load_embeddings(file):
    '''
    load embeddings txt file into a dictionary
    :param file:
    :return:
    '''
    embeddings_dictionary = dict()
    glove_file = open(file, encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    return embeddings_dictionary