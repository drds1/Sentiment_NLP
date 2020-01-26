import pandas as pd



if __name__ == '__main__':
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')



    '''some exploratory analysis'''

    # identify common key words
    keywords = test['keyword'].value_counts(dropna=False)
    print('most common keywords...\n',keywords.head(10))
    print('\nleast common keywords...\n',keywords.tail(10))
    print('\n\n\n')

    # identify common locations
    locations = test['location'].value_counts(dropna=False)
    print('most common locations...\n', locations.head(10))
    print('\nleast common locations...\n', locations.tail(10))
    print('\n\n\n')


