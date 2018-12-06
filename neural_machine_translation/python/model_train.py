from keras.callbacks import ModelCheckpoint

## Load own modules
## =========================================================

from lib import doc_processor
from lib import word_embedding
from lib import architecture

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
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))

# prepare german tokenizer
ger_tokenizer = word_embedding.create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = word_embedding.max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))

# prepare training data
trainX = word_embedding.encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = word_embedding.encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = word_embedding.encode_output(trainY, eng_vocab_size)

# prepare validation data
testX = word_embedding.encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = word_embedding.encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = word_embedding.encode_output(testY, eng_vocab_size)


## Define and train model
## =========================================================

# define model
model = architecture.define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# summarize defined model
print(model.summary())

# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)