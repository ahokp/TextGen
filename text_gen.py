import numpy as np

import keras
import keras.models
import keras.layers
import keras.callbacks
import keras.utils
import keras.utils.np_utils

from crawl_gutenberg import get_book_text_url, get_book_text
from preprocess_txt import tokenize_txts, lemmatize_words, get_sorted_vocabulary_and_word_count, \
                           prune_vocabulary


def get_lstm_model(vocab, lstm_layer_size=50, text_windowsize=10, features_per_input_sample=1):
    # Model params
    out_size = len(vocab)

    # Create the model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(lstm_layer_size, activation='relu', \
            input_shape=(text_windowsize, features_per_input_sample)))
    model.add(keras.layers.Dense(out_size, activation='softmax'))

    return model

def get_dataset(vocab, doc, text_windowsize=10):

    out_size = len(vocab)
    # Create dataset
    n_train_samples = len(doc) - text_windowsize
    x_data = np.zeros((n_train_samples, text_windowsize, 1))
    y_data = np.zeros((n_train_samples))

    for windowposition in range(n_train_samples):
        for k in range(text_windowsize):
            word = doc[windowposition+k]
            word_ind = np.where(vocab == str(word))[0][0]
            x_data[windowposition, k, 0] = word_ind
        
        word = doc[windowposition+text_windowsize]
        word_ind = np.where(vocab == str(word))[0][0]
        y_data[windowposition] = word_ind
    
    # One-Hot Encode
    y_ohe = keras.utils.np_utils.to_categorical(y_data, num_classes=out_size)

    return x_data, y_ohe

def generate_text(model, vocab, start_text, n_newtext=100, text_windowsize=10):
    # Starting text
    generated_wordindices = []
    for word in start_text:
        word_ind = np.where(vocab == word)[0][0]
        generated_wordindices.append(word_ind)

    # Generate new text to continue starting text
    for k in range(n_newtext):
        temp_textwindow = generated_wordindices[k: k + text_windowsize]
        temp_probs = model.predict(np.reshape(temp_textwindow, (1, text_windowsize, 1)))
        best_word = np.argmax(temp_probs)
        generated_wordindices = np.append(generated_wordindices, best_word)

    generated_text = vocab[generated_wordindices.astype(int)]
    return generated_text

def train(book_url):

    book_text_url = get_book_text_url(book_url)
    # Get text contents of book
    full_text_url = "https://www.gutenberg.org" + book_text_url
    book_text = [get_book_text(full_text_url)]

    # Tokenize texts and make all words lowercase
    tokenized_txts = tokenize_txts(book_text)

    # Lemmatize words
    lemmatized_texts = lemmatize_words(tokenized_txts)

    # Create a unified vocabulary of from the book text
    sorted_vocab, sorted_word_counts = get_sorted_vocabulary_and_word_count(lemmatized_texts)

    # Prune the vocabulary and texts
    pruned_text, sorted_pruned_vocab, sorted_pruned_counts = prune_vocabulary(lemmatized_texts, \
                                                                sorted_vocab, sorted_word_counts)

    ########### Generate text with lemmatized and vocab pruned text ##############

    # Get model
    model1 = get_lstm_model(sorted_pruned_vocab)
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Get data
    x_data1, y_ohe1 = get_dataset(sorted_pruned_vocab, pruned_text)

    # Fit data
    model1.fit(x=x_data1, y=y_ohe1, batch_size=128, epochs=30)

    # Generate text
    start_text1 = "enough talk leave prejudice room jane merely offered business money".split(" ")
    generated_text1 = generate_text(model1, sorted_pruned_vocab, start_text1)
    print("\nGenerated text with lemmatized and vocaulary-pruned text:")
    print(" ".join(w for w in generated_text1))
    print()

    ########## Generate text without lemmatization and vocab pruning ###########
    vocab2, word_counts2 = get_sorted_vocabulary_and_word_count(tokenized_txts)
    vocab2 = np.array(vocab2)

    # Get model
    model2 = get_lstm_model(vocab2)
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Get data
    x_data2, y_ohe2 = get_dataset(vocab2, tokenized_txts)

    # Fit data
    model2.fit(x=x_data2, y=y_ohe2, batch_size=256, epochs=70)

    # Generate text
    start_text2 = "enough to leave her room elizabeth was glad to be".split(" ")
    generated_text2 = generate_text(model2, vocab2, start_text2)
    print("\nGenerated text without lemmatization and vocabulary pruning:")
    print(" ".join(w for w in generated_text2))
    print()


if __name__ == "__main__":
    # Ebook URL to Pride and Prejudice by Jane Austen
    url = 'https://www.gutenberg.org/ebooks/1342'

    train(url)