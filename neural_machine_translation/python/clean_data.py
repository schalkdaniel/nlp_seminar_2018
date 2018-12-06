from numpy import array
from numpy.random import rand
from numpy.random import shuffle

## Load own modules
## =========================================================

from lib import doc_processor


## Clean test
## =========================================================

# load dataset
filename = 'data/eng_ger.txt'
doc = doc_processor.load_doc(filename)

# split into english-german pairs
pairs = doc_processor.to_pairs(doc)

# clean sentences
clean_pairs = doc_processor.clean_pairs(pairs)

# save clean pairs to file
doc_processor.save_clean_data(clean_pairs, 'data/english_german.pkl')

# spot check
for i in range(100):
  print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))


# Split into train and test
## =========================================================

# load clean dataset
raw_dataset = doc_processor.load_clean_sentences('data/english_german.pkl')

# reduce dataset size
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]

# define train-test fraction
train_fraction = 0.9
n_train = round(train_fraction * n_sentences)

# random shuffle
shuffle(dataset)

# split into train/test
train, test = dataset[:n_train], dataset[n_train:]

doc_processor.save_clean_data(dataset, 'data/english_german_both.pkl')
doc_processor.save_clean_data(train, 'data/english_german_train.pkl')
doc_processor.save_clean_data(test, 'data/english_german_test.pkl')

