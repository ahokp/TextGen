import nltk
import re

import numpy as np

def tokenize_txts(txts):
    nltk.download('punkt')
    # Tokenize book texts and add the words to a single list
    tokenized_txts = []
    
    for txt in txts:
        # Remove symbols from text
        txt = re.sub(r'[^\w]', ' ', txt)

        splitted_txt = nltk.word_tokenize(txt)
        tokenized_txts.extend(splitted_txt)

    return np.char.lower(np.array(tokenized_txts))

def lemmatize_words(word_list):
    nltk.download('wordnet')
    # Lemmatize words
    lemmatized_words = []

    lemmatizer = nltk.stem.WordNetLemmatizer()
    for word in word_list:
        lemmatized_words.append(lemmatizer.lemmatize(word))

    return lemmatized_words

def get_sorted_vocabulary_and_word_count(word_list):
    # Creates a vocabulary of the words and counts word occurances
    vocabulary = []
    word_counts = []

    for word in word_list:
        if word not in vocabulary:
            vocabulary.append(word)
            word_counts.append(1)
        else:
            i = vocabulary.index(word)
            word_counts[i] += 1

    # Sort according to word counts in descending order
    ind = np.argsort(-np.array(word_counts))
    sorted_vocab = np.array(vocabulary)[ind]
    sorted_word_counts = np.array(word_counts)[ind]

    return (sorted_vocab, sorted_word_counts)

def prune_vocabulary(lemmatized_texts, sorted_vocab, sorted_word_counts):

    words_to_remove = []
    nltk.download('stopwords')
    nltkstopwords = nltk.corpus.stopwords.words('english')

    # Remove top 1% most frequent words
    top1_percent = int(len(sorted_vocab)*0.01)
    words_to_remove.extend(sorted_vocab[:top1_percent])
    sorted_vocab = sorted_vocab[top1_percent:]
    sorted_word_counts = sorted_word_counts[top1_percent:]

    # Remove words occurring less than 4 times
    more_than_3 = np.where(sorted_word_counts > 3)[0]
    words_to_remove.extend(sorted_vocab[np.where(sorted_word_counts <= 3)[0]])
    sorted_vocab = sorted_vocab[more_than_3]
    sorted_word_counts = sorted_word_counts[more_than_3]

    # Remove stopwords downloaded from nltk
    stopword_indices = np.where(np.in1d(sorted_vocab, nltkstopwords))[0]
    words_to_remove.extend(sorted_vocab[stopword_indices])
    sorted_vocab = np.delete(sorted_vocab, stopword_indices)
    sorted_word_counts = np.delete(sorted_word_counts, stopword_indices)

    # Remove overly short words
    ind_too_short = np.where(np.char.str_len(sorted_vocab) < 3)[0]
    words_to_remove.extend(sorted_vocab[ind_too_short])
    sorted_vocab = np.delete(sorted_vocab, ind_too_short)
    sorted_word_counts = np.delete(sorted_word_counts, ind_too_short)

    # Remove word if it's too long
    ind_too_long = np.where(np.char.str_len(sorted_vocab) > 20)[0]
    words_to_remove.extend(sorted_vocab[ind_too_long])
    sorted_vocab = np.delete(sorted_vocab, ind_too_long)
    sorted_word_counts = np.delete(sorted_word_counts, ind_too_long)

    pruned_text = []
    # Prune text
    for word in lemmatized_texts:
        if word not in words_to_remove:
            pruned_text.append(word)

    return pruned_text, sorted_vocab, sorted_word_counts
