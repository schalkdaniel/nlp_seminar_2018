import re
from numpy import array
from keras.models import load_model

## Load own modules
## =========================================================

from lib import doc_processor
from lib import word_embedding
from lib import evaluation


## Load data
## =========================================================

# Load clean data created by 'clean_data.py'
dataset = doc_processor.load_clean_sentences('data/english_german_both.pkl')
train = doc_processor.load_clean_sentences('data/english_german_train.pkl')
test = doc_processor.load_clean_sentences('data/english_german_test.pkl')

## Create train and test word embeddings
## =========================================================

# prepare english tokenizer
eng_tokenizer = word_embedding.create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = word_embedding.max_length(dataset[:, 0])

# prepare german tokenizer
ger_tokenizer = word_embedding.create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = word_embedding.max_length(dataset[:, 1])

# prepare data
trainX = word_embedding.encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testX = word_embedding.encode_sequences(ger_tokenizer, ger_length, test[:, 1])

## Load model
## =========================================================

# load model trained by 'model_train.py'
model = load_model('model.h5')

# test on some training sequences
evaluation.evaluate_model(model, eng_tokenizer, trainX, train)

# test on some test sequences
evaluation.evaluate_model(model, eng_tokenizer, testX, test)

# translate arbitrary sentence

my_sentence = 'ich gehe heute abend essen'
my_sentence_we = word_embedding.encode_sequences(ger_tokenizer, ger_length, array([my_sentence]))

for word in re.split('\s+', my_sentence):
  print('word=[%s], integer=[%s]' % (word, ger_tokenizer.word_index.get(word)))

my_sentence_we

evaluation.predict_sequence(model, eng_tokenizer, my_sentence_we)