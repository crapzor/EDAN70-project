"""
Trains the model and predicts the chunk-tags. Saves result into files.
"""

from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from pathlib import Path

import numpy as np
import tensorflow as tf

# Constants
EMBEDDING_DIM = 100
GLOVE_LENGTH = 400000
np.random.seed(1)


# dictionary of words from the glove.6b, one with
# the words as keys with their vector as values (e_idx)
# and the other with the words as keys their index
# as values (w_idx)
def create_embedding(file, e_idx, w_idx):
    f = open(file)
    idx = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        e_idx[word] = coefs
        idx += 1
        w_idx[word] = idx
    f.close()


# read sentences from file and return
def read_sentences(file):
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


# take out word's form and chunk, save in a list
# of dictionaries, also take out the longest sentence
def split_rows(sentences):
    all_sentences = []
    longest = 0
    dict_sentence = {'form': [], 'chunk': []}
    for sentence in sentences:
        rows = sentence.split('\n')
        for row in rows:
            r = row.split()
            dict_sentence['form'].append(r[0].lower())
            dict_sentence['chunk'].append(r[-1])
        all_sentences.append(dict_sentence)
        if len(dict_sentence['form']) > longest:
            longest = len(dict_sentence['form'])
        dict_sentence = {'form': [], 'chunk': []}
    return all_sentences, longest


# adds the words from the train file, that doesn't
# exist in GloVe, to the dictionary of words-indexes
def add_words_from_train_file(word_index, train_dictionary, LENGTH):
    idx = LENGTH
    for sentence in train_dictionary:
        for word in sentence['form']:
            if word not in word_index:
                idx += 1
                word_index[word] = idx
    return idx

# create the embedding matrix, first row is zeros (0), then
# all the rows from the glove embeddings, then random vectors
# for the words from the training data that doesn't exist in
# glove, and lastly the random row for all the unknown
def create_embedding_matrix(word_idx, embedding_index, LENGTH):
    em_matrix = np.zeros((LENGTH + 2, EMBEDDING_DIM))
    em_matrix[LENGTH + 1] = np.random.rand(1,100) * 2 - 1 # last row is random
    for word, i in word_idx.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            em_matrix[i] = embedding_vector
        else:
            em_matrix[i] = np.random.rand(1, 100) * 2 - 1
    return em_matrix


# pads each sentence with zeros so that they all are of
# equal lengths
def sentence_padder(sentences, word_idx, chunk_idx, length, LENGTH):
    form_idx_list = list()
    chunk_idx_list = list()
    for sentence in sentences:
        padded_form = [0] * (length - len(sentence['form']))
        padded_chunk = [0] * (length - len(sentence['form']))
        for i in range(len(sentence['form'])):
            if word_idx.get(sentence['form'][i]) is None:
                # if word doesn't exist in our list of words
                # we give it the label "unkown"
                padded_form.append(LENGTH + 1)
            else:
                padded_form.append(word_idx[sentence['form'][i]])
            padded_chunk.append(chunk_idx[sentence['chunk'][i]])
        form_idx_list.append(padded_form)
        chunk_idx_list.append(padded_chunk)
    return np.array(form_idx_list), np.array(chunk_idx_list)


# retrieves the different chunk-tags in the training data
def get_chunks(sentence_dictionary):
    chunk_dict = dict()
    idx = 0
    for sentence in sentence_dictionary:
        for i in range(len(sentence['chunk'])):
            if sentence['chunk'][i] not in chunk_dict:
                idx += 1
                chunk_dict[sentence['chunk'][i]] = idx
    return chunk_dict


# extracts the necessary data from the output
def extract_useful_data(raw_data, dictionary, longest):
    data = list()
    for i in range(raw_data.shape[0]):
        sentence_length = len(dictionary[i]['chunk'])
        data.append(raw_data[i][longest - sentence_length:longest])
    return data


# saves into file the predicted chunks
def save_to_file(file_name, sentences, chunk_list, predicted):
    f_out = open(file_name, 'w')
    for i in range(len(sentences)):
        rows = sentences[i].splitlines()
        for j in range(len(rows)):
            row = rows[j] + ' ' + chunk_list[predicted[i][j] - 1]
            f_out.write(row + '\n')
        f_out.write('\n')
    f_out.close()


if __name__ == '__main__':
    embeddings_index = dict()
    word_index = dict()

    model_name = "english.model"
    train = False
    train_file = 'corpus/conv_eng.train'
    testa_file = 'corpus/conv_eng.testa'
    testb_file = 'corpus/conv_eng.testb'
    output_file_a = 'predicted_eng.testa'
    output_file_b = 'predicted_eng.testb'
    glove_file = 'glove.6B/glove.6B.100d.txt'

    # getting the embedding matrix
    create_embedding(glove_file, embeddings_index, word_index)
    # sentences, dictionary of sentences, and longest sentence of training data
    train_sentences = read_sentences(train_file)
    train_dictionary, longest_sentence_train = split_rows(train_sentences)

    # complement word_index with what's missing from train file
    WORD_INDEX_LENGTH = add_words_from_train_file(word_index, train_dictionary, GLOVE_LENGTH)

    embedding_matrix = create_embedding_matrix(word_index, embeddings_index, WORD_INDEX_LENGTH)
    # sentences, dictionary of sentences, and longest sentence of test A data
    testa_sentences = read_sentences(testa_file)
    testa_dictionary, longest_sentence_testa = split_rows(testa_sentences)
    # sentences, dictionary of sentences, and longest sentence of test B data
    testb_sentences = read_sentences(testb_file)
    testb_dictionary, longest_sentence_testb = split_rows(testb_sentences)
    # longest sentence in order to know how much to pad etc
    longest_sentence = max(longest_sentence_train, longest_sentence_testa, longest_sentence_testb)

    # dictionary of the chunks and their respective indices
    chunk_index = get_chunks(train_dictionary)
    # list of the different types of chunks
    chunk_list = list(chunk_index.keys())

    # padding the train sentences
    form_idx_train, chunk_idx_train = sentence_padder(train_dictionary, word_index, chunk_index, longest_sentence, WORD_INDEX_LENGTH)

    training_samples = form_idx_train.shape[0]

    indices_train = np.arange(form_idx_train.shape[0])
    forms_train = form_idx_train[indices_train]
    chunks_train = chunk_idx_train[indices_train]

    # one-hot encode the chunk-tags
    y_train = list()
    for i in chunks_train:
        y_train.append(utils.to_categorical(i, num_classes=10))

    x_train = forms_train[:training_samples]
    y_train = np.array(y_train)

    # if model already exists - get it
    # otherwise train it
    my_model = Path(model_name)
    if my_model.is_file():
        print("Loading model...")
        model = tf.keras.models.load_model(model_name)
    else:
        print("Training model...")
        model = Sequential()
        model.add(Embedding(WORD_INDEX_LENGTH + 2, EMBEDDING_DIM,
                            mask_zero=True, weights=[embedding_matrix],
                            input_length=longest_sentence, trainable=train))
        model.add(Bidirectional(LSTM(units=EMBEDDING_DIM, dropout=0.5, return_sequences=True)))
        model.add(Bidirectional(LSTM(units=EMBEDDING_DIM, return_sequences=True)))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(optimizer='rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['acc'])
        checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
        model.fit(x_train, y_train,
                            epochs=20,
                            batch_size=32,
                            validation_split=0.1,
                            callbacks=[checkpointer])
        model.save(model_name)

    print(model.summary())


    # test the test A data
    form_idx_testa, chunk_idx_testa = sentence_padder(testa_dictionary, word_index, chunk_index, longest_sentence, WORD_INDEX_LENGTH)
    indices_testa = np.arange(form_idx_testa.shape[0])
    forms_testa = form_idx_testa[indices_testa]
    chunks_testa = chunk_idx_testa[indices_testa]

    y_testa = list()
    for i in chunks_testa:
        y_testa.append(utils.to_categorical(i, num_classes=10))

    x_testa = form_idx_testa
    y_testa = np.array(y_testa)
    # predict the data
    raw_predicted_testa = model.predict_classes([x_testa])

    # extracts the necessary data from the raw data
    predicted_testa = extract_useful_data(raw_predicted_testa, testa_dictionary, longest_sentence)
    # save to file
    save_to_file(output_file_a, testa_sentences, chunk_list, predicted_testa)


    # test the test B data
    form_idx_testb, chunk_idx_testb = sentence_padder(testb_dictionary, word_index, chunk_index, longest_sentence, WORD_INDEX_LENGTH)
    indices_testb = np.arange(form_idx_testb.shape[0])
    forms_testb = form_idx_testb[indices_testb]
    chunks_testb = chunk_idx_testb[indices_testb]

    y_testb = list()
    for i in chunks_testb:
        y_testb.append(utils.to_categorical(i, num_classes=10))
    x_testb = form_idx_testb
    y_testb = np.array(y_testb)
    #predict the data
    raw_predicted_testb = model.predict_classes([x_testb])

    #extracts the necessary data from the raw data
    predicted_testb = extract_useful_data(raw_predicted_testb, testb_dictionary, longest_sentence)
    # save to file
    save_to_file(output_file_b, testb_sentences, chunk_list, predicted_testb)
