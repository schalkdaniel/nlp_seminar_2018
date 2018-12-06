# Neural Machine Translation

## Run the Code

The code used here is for the most part copied from [this post](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/). 

1. run `python/clean_data.py`
1. run `python/model_train.py`
1. run `python/model_evaluate.py`

## Interesting Tasks

- [ ] What are the challenges in neural machine translation?

- [ ] How does the encoder-decoder model tackle these challenges?
    - [ ] Why are we using RNNs and LSTMs?
    - [ ] How flexible are we in extending the architecture (including more layers)?

- [ ] How to evaluate a neural net for machine translation?
    - [ ] BLEU score
    - [ ] Other performance measures?

- [ ] How does the used dictionary affect the prediction?
    - [ ] What sources are available?
    - [ ] How much data are necessary to train a reasonably useful model?

- [ ] **[EXTRA]** How does word embeddings affect the predictions?
- [ ] **[EXTRA]** Available translation tools and what are they using (e.g. Google, DeepL, ...)?
- [ ] **[EXTRA]** How much computational power do we need to get a reasonably useful model?

## Data Sources

- [Tatoeba dataset](data/_about_tatoeba.md)