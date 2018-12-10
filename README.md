# Tweets-Sentiment-Analysis

## Goal

The purpose of this project is to give a score of sentiment to a tweet using NLP methods and neural networks. This score can take 7 values defined as follow:

- 3: very positive emotional
- 2: moderately positive emotional
- 1: slightly positive emotional
- 0: neutral or mixed emotional
- -1: slightly negative emotional
- -2: moderately negative emotional
- -3: very negative emotional

For example, 

- a score of `3` should be return with a tweet like `"I found #marmite in Australia. :) #happy"`,
- a score of `-3` should be return with a tweet like `"I hate having ideas but being too afraid to share them 😔"`.

The main issue is that there are very few tweets labeled at 7, but on the other hand there are a lot of tweets labeled at 3. So, a part of this project is to apply a transfere learning with a neural network trained with tweets labeled at 3 onto another neural network for tweets labeled at 7.

Others part of the project is to preprocess the tweets, produce word embeddings to have a strong representations of the language, add some data sources like `Emoji Valence` or `Opinion Lexicon English` sources in ordre to have more semantics in our vectors.

## Resume

This program has been developed with Python programming language.

The project contains:

* src folder containing the source files
* data folder containing
     * datasets for tweets
     * embedding matrix
* resources folder containing
    * EV source (emoji valence)
    * OLE source (opinion lexicon english)
* results where models and predictions are produced

## Installation

Go to `data/embedding/README.md` and follow the instruction.

## Execution

Run the Notebook `src/train.ipynb` into Jupyter and launch the cells. The results are produced into `results/`.

## Bibliography

Some internet sources that help me to understand the problem and propose a solution:

- tweets preprocessing: https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/
- bag of words: https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4855001-representez-votre-corpus-en-bag-of-words
- word embeddings: https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4855006-effectuez-des-plongements-de-mots-word-embeddings
- create its own word-embeddings: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb
- natural network LSTM: https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras
- Emoji sentiment ranking: http://kt.ijs.si/data/Emoji_sentiment_ranking/about.html

## Authors

EPITA School, SCIA Master 2 - Project for Deep Learning and Natural Language Processing Course. 

Authors: 
- **BENSABAT David** (bensab_d)
- **YVONNE Xavier** (xavier.yvonne)
