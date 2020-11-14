# -*- coding: utf-8 -*-
"""Text processing

Simplified version of a text processing pipeline.

    * load data
    * clean text
    * vectorize (with a word embedding)
    * reduce dimension
    * cluster

Sources:
    https://www.kaggle.com/tmdb/tmdb-movie-metadata
    https://fasttext.cc/docs/en/pretrained-vectors.html
    https://radimrehurek.com/gensim_3.8.3/models/fasttext.html
    https://radimrehurek.com/gensim_3.8.3/auto_examples/tutorials/run_fasttext.html

Pre-trained FastText:
    https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
"""
__version__ = "0.1.0"

import configparser
import json
import logging
import functools
import os
import string

import gensim
import hdbscan
import nltk
import numpy as np
import pandas as pd
import sklearn

CONFIG_FILEPATH = os.path.join(
    os.path.expanduser("~"), "text_processing_config.ini"
)


def _read_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILEPATH)
    return config


def _setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def fetch_stopwords():
    nltk.download("stopwords")
    stopwords = nltk.corpus.stopwords.words("english")
    stopword_set = set(word.replace("'", "") for word in stopwords)
    return stopword_set


config = _read_config()
logger = _setup_logging()

STOPWORDS = fetch_stopwords()
RANDOM_STATE = 3052528580

def fetch_files():
    CPA = pd.read_csv("../data/processed/CPA_data_cleaned.csv")
    logger.info("cleanded CPA File imported")
    # Create categories for Level 1 and Level 2 higherarchies
    CPA2 = CPA.copy()
    #get highest level of code
    CPA2.loc[CPA2.Level !=1,'Category_2'] = CPA2[CPA2.Level !=1].Code.str.split('.').str.slice(0,1).str.join('')

    # match up codes and parents
    Code_parent = CPA2[CPA2.Level==2][['Parent','Category_2']].copy()
    CPA2 = CPA2.merge(Code_parent.rename(columns={'Parent':'Category_1'}), on='Category_2', how='left')

    # add in the Category_1 fileds for Level1
    CPA2.loc[CPA2.Level==1, 'Category_1'] = CPA2.loc[CPA2.Level==1,'Code']

    # we now set up a higher level for A10 indstry levels (10 categories)
    update_dict0 = {'A':1,'F':3,'J':5, 'K':6, 'L':7,'M':8,'N':8}
    update_dict = {**update_dict0,**dict.fromkeys(['B','C','D','E'],2),**dict.fromkeys(['G','H','I'],4),
                   **dict.fromkeys(['O','P','Q'],9), **dict.fromkeys(['R','S','T','U'],10)}

    CPA2['Category_0'] = CPA2.Category_1.replace(update_dict)

    CPA_L6 = CPA2[CPA2.Level==6].copy()
    
    # join the L5 description to the L6
    CPA_L5 = CPA2[CPA2.Level==5].copy()

    CPA_L56 = CPA_L6.merge(CPA_L5[['Code','Descr']].rename(columns={'Descr':'L5_Descr','Code':'Code_L5'}), left_on='Parent', right_on='Code_L5', how='left')
    CPA_L56['Full_descr'] = CPA_L56['L5_Descr'].fillna('')+ ' ' + CPA_L56['Descr'].fillna('')
    CPA_L56 = CPA_L56.drop(['Code_L5','Parent','L5_Descr'],axis=1)

    return CPA_L56



def hello_you():
    print('hello you!')


def clean_text(text: str):
    """Prepare overview text for further processing.

    * remove punctuation
    * lowercase
    * no stopwords
    """
    text = text.replace("!", ".").replace("?", ".")
    all_punc = string.punctuation
    all_punc = all_punc.replace('-','')
    translation_table = str.maketrans("", "", all_punc)
    no_punctuation = text.translate(translation_table)

    lowercase = no_punctuation.lower()

    no_stopwords = " ".join(
        word for word in lowercase.split(" ") if word not in STOPWORDS
    )

    return no_stopwords


def clean_col(df: pd.DataFrame, col: str):
    """Prepare column for text processing.

    Returns:
        Same dataframe with an additional column for cleaned text.
    """
    logger.info(f"Cleaning column: {col} ")
    df[f"{col}_cleaned"] = df[col].fillna("").apply(clean_text)
    return df


def fetch_fasstext_pretrained(filepath=None):
    if filepath is None:
        filepath = config["filepaths"]["FastTextPretrainedBinary"]

    logger.info(f"Loading FastText pretrained from {filepath}")
    wv = gensim.models.fasttext.load_facebook_vectors(filepath)

    logger.info("Model loaded")
    return wv


def vectorize_text(
    wv: gensim.models.keyedvectors.WordEmbeddingsKeyedVectors, text: str
):
    """Apply word vectorizer to text.

    This takes a simple averaging approach
    i.e. every word in the text is passed to the model and the resulting
    vectors are averaged.
    """
    
    vecs = np.array([wv[word] for word in text.split(" ")])

    return np.mean(vecs, axis=0)


def reduce_dimensionality(vector_col: pd.Series):
    # There have been issues with the umap import
    import umap

    logger.info("Applying umap to reduce dimension")
    vecs = np.array(list(vector_col.values))

    clusterable_embedding = umap.UMAP(
        n_neighbors=10,
        min_dist=0.0,
        n_components=10,
        random_state=RANDOM_STATE,
        verbose=10,
    ).fit_transform(vecs)

    return pd.Series(data=clusterable_embedding.tolist(), index=vector_col.index)


def cluster(vector_col: pd.Series):

    vecs = np.array(list(vector_col))

    labels = hdbscan.HDBSCAN().fit_predict(vecs)

    return pd.Series(data=labels.tolist(), index=vector_col.index)


def evaluate(df, original, labels):
    """Cluster similarity"""
    labels_true = df[original]
    labels_pred = df[labels]
    result = sklearn.metrics.cluster.adjusted_rand_score(
        labels_true=labels_true, labels_pred=labels_pred
    )

    logger.info("Are these labellings anything like each other?")
    logger.info(f"Adjusted rand score: {result}")


def plot_embedding(df):
    import matplotlib.pyplot as plt
    import seaborn
    import sklearn

    target = sklearn.preprocessing.LabelEncoder().fit_transform(df.genres)

    vecs = np.array(list(df["overview_cleaned_vectorized_low_dimension"].values))

    plt.scatter(vecs[:, 0], vecs[:, 1], c=target, s=0.1, cmap="Spectral")

    plt.show()


if __name__ == "__main__":

    hello_you()
    
