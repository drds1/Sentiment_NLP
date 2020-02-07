import prediction_notebook as pn
import glob


def load_imdb_data(path_to_files = './extra_data/aclImdb/train/neg/',
                   extension = '*.txt'):
    '''

    :param path_to_files:
    :param extension:
    :return:
    '''
    files = glob.glob(path_to_files+extension)
    text = []
    for f in files:
        with open(f, "r") as myfile:
            text.append(myfile.read().replace('\n', ''))
    return text



if __name__ == '__main__':


    #ingest the negative data
    text_neg = load_imdb_data(path_to_files='./extra_data/aclImdb/train/neg/',extension='*.txt')
    text_pos = load_imdb_data(path_to_files='./extra_data/aclImdb/train/pos/', extension='*.txt')
    text_train = list(text_neg) + list(text_pos)
    label_train = [0]*len(text_neg) + [1]*len(text_pos)

    text_neg = load_imdb_data(path_to_files='./extra_data/aclImdb/test/neg/', extension='*.txt')
    text_pos = load_imdb_data(path_to_files='./extra_data/aclImdb/test/pos/', extension='*.txt')
    text_test = list(text_neg) + list(text_pos)
    label_test = [0] * len(text_neg) + [1] * len(text_pos)

    y = pn.Nlp_General()
    y.epochs = 2
    y.validation_split = 0.2
    y.batch_size = 64
    y.load_data(train_text=text_train, train_target=label_train,
                test_text=text_test,
                glove_file='./data/glove.6B.100d.txt')
