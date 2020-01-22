import pandas as pd, spacy, random, pickle
import numpy as np, os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer

def gen_data_frames(df, genre2int, max_num):

    nr_training_examples = {'Rock'   : 0,
                        'Pop'        : 0,
                        'Metal'      : 0,
                        'Country'    : 0,
                        'Hip-Hop'    : 0,
                        }

    '''nr_training_examples = {'Rock'   : 0,
                        'Pop'        : 0,
                        'R&B'        : 0,
                        'Metal'      : 0,
                        'Country'    : 0,
                        'Folk'       : 0,
                        'Hip-Hop'    : 0,
                        'Indie'      : 0,
                        'Jazz'       : 0,
                        'Electronic' : 0
                        }'''

    index = []
    lyrics = []
    labels = []
    index = 0
    list_of_genres = list(nr_training_examples.keys())
    for i in df['genre']:

        if i in list_of_genres and nr_training_examples[i] != max_num:
            nr_training_examples[i] += 1
            lyrics.append(df['lyrics'][index])
            labels.append(genre2int[i])
            index += 1
        else:
            index += 1
            continue

    df = pd.DataFrame(data=zip(lyrics,labels), columns=['lyrics', 'genre'])

    x = lyrics
    y = labels
    return df, x, y

def get_stopwords(str_stopwords, punctuation):
    str_stopwords += ['/n']
    str_stopwords += punctuation
    return str_stopwords

def tokenize(text_data, stopwords):
    training_data = []
    index = 0
    tokenized = []

    for i in tqdm(text_data):

        tokens = word_tokenize(i.lower())
        for x in tokens:
            if x in stopwords:
                tokens.remove(x)

            else:
                continue
        tokenized.append(tokens)

    return tokenized

def lemmatized(text_data):
    nlp = spacy.load('en_core_web_sm')
    lemmatized_data = []
    for i in tqdm(text_data):

        doc = nlp(' '.join(i))
        lemmatized = []
        for word in doc:
            lemmatized.append(word.lemma_)
        lemmatized_data.append(lemmatized)
    return lemmatized_data

def output_csv(data, labels, path):
    untokenized = []
    for i in data:
        untokenized.append(' '.join(i))
    df = pd.DataFrame(data=zip(untokenized, labels), columns=['lyrics', 'genre'])
    df.to_csv(path)
    return df

def output_txt(x, y, x_path, y_path):

    directory = 'data/pre_processed/'

    if os.path.exists(directory) == False:
        os.mkdir(directory)

    with open('{}{}'.format(directory, x_path), 'w') as file:
        for i in x:
            file.write(' '.join(i))
            file.write('\n')

    with open('{}{}'.format(directory,y_path), 'w') as file:
        for i in y:
            file.write(str(i))
            file.write('\n')

def output_dl(data, filename):
    with open('dataloaders/' + filename, 'wb') as file:
        pickle.dump(data, file)

def main():

    print('Loading lyric csv')
    train_lyrics = pd.read_csv('data/raw/train.csv')
    test_lyrics = pd.read_csv('data/raw/test.csv')
    genres = ['Rock', 'Pop', 'Metal', 'Country',
      'Hip-Hop']
    '''genres = ['Rock', 'Pop', 'R&B', 'Metal', 'Country', 'Folk',
      'Hip-Hop', 'Indie', 'Jazz', 'Electronic']'''
    int2genre = dict(enumerate(genres))
    genre2int = {genre : num for num, genre in int2genre.items()}
    print('Trimming datast')
    training_df, x_train, y_train = gen_data_frames(train_lyrics, genre2int, 5000)
    testing_df, x_test, y_test = gen_data_frames(test_lyrics, genre2int, 5000)
    stops = get_stopwords(stopwords.words('english'), punctuation)
    print('Tokenizing')
    train_tokenized = tokenize(x_train, stops)
    output_txt(train_tokenized, y_train, 'x_train.txt', 'y_train.txt')
    test_tokenized = tokenize(x_test, stops)
    output_txt(test_tokenized, y_test, 'x_test.txt', 'y_test.txt')
    print('Lemmatizing')
    train_lemm = lemmatized(train_tokenized)
    output_txt(train_lemm, y_train, 'x_train_lemm.txt', 'y_train_lemm.txt')
    test_lemm = lemmatized(test_tokenized)
    output_txt(test_lemm, y_test, 'x_test_lemm.txt', 'y_test_lemm.txt')

if __name__ == '__main__':
    main()
