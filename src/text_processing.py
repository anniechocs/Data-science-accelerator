# -*- coding: utf-8 -*-
"""Text processing

Simplified version of a text processing pipeline.

    * load data
    * clean text
    * vectorize (with a word embedding)
    * reduce dimension
    * cluster

Sources:l
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

import file_import_helper

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
    # The CPA codes have already been cleansed and are now loaded and split into categories
    
    data_path = "../data/processed/"
    CPA = pd.read_csv(data_path + "CPA21_cleaned.csv")
    CPA.columns = [x[4:] for x in CPA.columns]
    logger.info("cleanded CPA File imported")
    CPA = CPA.rename(columns={'Description':'Descr','Description_old':'Descr_old'}).copy()
    
    CPA2 = file_import_helper.CPA_Categories(CPA)

    CPA2.to_csv(data_path + 'CPA_cleaned.csv', index=False)
    return CPA2

def fetch_CN_files():
    import exclude_remove
    data_path = '../data/raw/'

    #### read in and process CN 2010 file
    CN10 = pd.read_excel(data_path + "CN_2010_raw.xls",  skiprows=1,names=['Key','CN_Code', 'Descr'], usecols=[0,1,2], dtype={'Key':str})
    # read in and merge structure data so we can get the levels
    CN10st = pd.read_excel(data_path + "CN 2010_structure.xls", usecols=[0,1,2], names=['Key','CN_Code','Level'],  dtype={'Key':str})
    CN10 = CN10.merge(CN10st[['Key','Level']], on='Key',how='left')
    CN10 = CN10.sort_values(by='Key')
    CN10['CN_Code'] = CN10.CN_Code.str.replace(' ','')
    # clean CN10 data
    CN10['Exclusions_removed'] = ''
    CN10 = exclude_remove.remove_exclusions(CN10, 'Descr', 'Exclusions_removed')
    # select the columns we want
    CN10=CN10[CN10.Descr_cleaned.notnull()][['Key','CN_Code','Level','Descr_old','Descr_cleaned', 'Exclusions_removed']].rename(columns={'Descr_cleaned':'Descr'})

    # read in CN 2010 file - we have a full description column already here
    CN20 = pd.read_csv(data_path + "CN_2020_raw.csv", header=None, sep=',', skiprows=1, usecols=[0,1,2,3,4,5,6,7],
            names=['Order','Level','Key','Parent','CN_Code','CN_Parent','Descrip','Descr'],  dtype={'Key':str, 'Parent':str})
    # clean CN20 data
    CN20['CN_Code'] = CN20.CN_Code.str.replace(' ','')
    CN20 = exclude_remove.remove_exclusions(CN20, 'Descr', 'Exclusions_removed')
    # select the columns we want
    CN20=CN20[CN20.Descr_cleaned.notnull()][['Key','CN_Code','Level','Descr_old','Descr_cleaned', 'Exclusions_removed']].rename(columns={'Descr_cleaned':'Descr'})

    return CN10, CN20

def fetch_CN_mapper():
    import exclude_remove
    data_path = '../data/processed/'

    CN = pd.read_csv(data_path+"CN_2020_full.csv", header=None, sep=',', names=['CN_Code','CN_Section','CN_Description'])
    CN['CN_Code'] = np.floor_divide(CN['CN_Code'],10000)
    CN['CN_Code'] = CN.CN_Code.astype(int)

    # read in mapping data
    CN_CPA = pd.read_csv(data_path+'CN 2020-CPA 2.1.csv', skiprows=1)
    CN_CPA['Source'] = pd.to_numeric(CN_CPA['Source'].str.replace(' ',''))
    
    CC0 = CN_CPA.rename(columns={'Source':'CN_Code', 'Target':'CPA_Code'}).merge(CN, on='CN_Code',  how='right')
    CC0['CN_Level'] = CC0.CN_Section.str.len()
    
    CC = exclude_remove.remove_exclusions(CC0,'CN_Description','Excl_removed')
    
    CC.loc[CC.CPA_Code.notnull(),'Category_2'] = CC[CC.CPA_Code.notnull()].CPA_Code.str.split('.').str.slice(0,1).str.join('')
    
    
    logger.info("CN new Files imported and cleaned")
    
    return CC
    

def get_category_description(level_wanted):
    CPA = fetch_files()
    cat = 'Category_' + str(level_wanted)
    if cat == 'Category_0':
        Cat_0_dict  = {'1':'Agriculture (A)', '2':'Production (B-E)', '3':'Construction (F)', '4':'Trade, Transport, Hospitality (G-I)',
            '5':'Information & Communication (J)', '6':'Financial & Insurance (K)', '7':'Real Estate (L)',
             '8':'Scientific, Technical and Admin Services (M-N)','9':'Education, Health, Social Work (O-Q)', '10':'Other Services (R-U)'}
        Cat_0_descrip = pd.DataFrame.from_dict(Cat_0_dict, orient ='index').reset_index()
        Cat_0_descrip.columns =['Category_0','Cat_0_Description']
        tmp = Cat_0_descrip
    elif cat == 'Category_1':
        Cat_1_descrip = CPA[CPA.Level==1][['Category_1','Descr']].rename(columns={'Descr':'Cat_1_Description'})
        Cat_1_descrip['Cat_1_Description'] = Cat_1_descrip.Cat_1_Description.str.title()
        Cat_1_descrip['Cat_1_Description'] = Cat_1_descrip['Cat_1_Description'].replace('And','and', regex=True)
        Cat_1_descrip['Cat_1_Description'] = Cat_1_descrip['Cat_1_Description'].where(Cat_1_descrip['Category_1']!='T','Services Of Households As Employers etc')
        Cat_1_descrip['Cat_1_Description'] = Cat_1_descrip['Cat_1_Description'].where(Cat_1_descrip['Category_1']!='G','Wholesale and Retail Trade Services, Motor Services')
        Cat_1_descrip['Cat_1_Description'] = Cat_1_descrip['Cat_1_Description'].where(Cat_1_descrip['Category_1']!='O','Public Administration, Defences, Social Security Services')
        tmp = Cat_1_descrip
    else:    
        tmp = CPA[CPA.Level==level_wanted][['Code','Descr_old']].rename(columns={'Code':cat, 'Descr_old':cat+'_descr'})
    return tmp
    
def get_levels_5_6(CPA):    

    CPA_L6 = CPA[CPA.Level==6].copy()
    
    # join the L5 description to the L6
    CPA_L5 = CPA[CPA.Level==5].copy()

    CPA_L56 = CPA_L6.merge(CPA_L5[['Code','Descr']].rename(columns={'Descr':'L5_Descr','Code':'Code_L5'}), left_on='Parent', right_on='Code_L5', how='left')
    CPA_L56['Full_descr'] = CPA_L56['L5_Descr'].fillna('')+ ' ' + CPA_L56['Descr'].fillna('')
    CPA_L56 = CPA_L56.drop(['Code_L5','Parent','L5_Descr'],axis=1)

    return CPA_L56


def hello_you():
    print('hello you!!!!!')


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


def clean_col(df: pd.DataFrame, col: string):
    """Prepare columns for text processing.

    Returns:
        Same dataframe with an additional column for cleaned text.
    """
    logger.info(f"Cleaning column: {col} ")
    df[f"{col}_cleaned"] = df[col].fillna("").apply(clean_text)
    return df

def clean_cols(df: pd.DataFrame, cols: list):
    """Prepare columns for text processing.

    Returns:
        Same dataframe with an additional columns for cleaned text.
    """
    for col in cols:
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


def reduce_dimensionality(vector_col: pd.Series, n_comp: int):
    # There have been issues with the umap import
    import umap

    logger.info("Now applying umap to reduce dimension")
    vecs = np.array(list(vector_col.values))

    clusterable_embedding = umap.UMAP(
        n_neighbors=10,
        min_dist=0.0,
        n_components=n_comp,
        random_state=RANDOM_STATE,
        verbose=10,
    ).fit_transform(vecs)

    return pd.Series(data=clusterable_embedding.tolist(), index=vector_col.index)

def reduce_dimensionality_supervised(vector_col: pd.Series, target_col: pd.Series):
    # There have been issues with the umap import
    import umap

    logger.info("Now applying umap to reduce dimension")
    vecs = np.array(list(vector_col.values))
    target = np.array(list(target_col.values))
    
    clusterable_embedding = umap.UMAP(
        n_neighbors=15,  # larger number suggested for supervised classification.
        min_dist=0.0,
        n_components=10,
        random_state=RANDOM_STATE,
        verbose=10,
    ).fit_transform(vecs, y=target)

    return pd.Series(data=clusterable_embedding.tolist(), index=vector_col.index)

def train_test_umap(X_train: pd.Series, y_train: pd.Series, X_test: pd.Series):
    # There have been issues with the umap import
    import umap

    logger.info("Now applying umap to reduce dimension")
    vecs = np.array(list(X_train.values))
    target = np.array(list(y_train.values))
    test = np.array(list(X_test.values))
    
    mapper = umap.UMAP(n_neighbors=10,
                        min_dist=0.0,
                        n_components=10,
                        random_state=RANDOM_STATE,
                        verbose=10,                     
                      ).fit(vecs,target)   

    
    test_embedding = mapper.transform(test)
    # add embedding to the train set  
    return pd.Series(data=test_embedding.tolist(), index=X_test.index)

def plot_scat(df, cat, plt_title):
    import matplotlib.pyplot as plt
    import seaborn as sns
    x_col = df.columns[-2]
    y_col = df.columns[-1]
    
    # get a dataframe with descriptions for the category level we are investigating
    descr_df = get_category_description(cat)
    categ = descr_df.columns[0]
    descr = descr_df.columns[1]
    df1 = df.merge(descr_df, on = categ,  how='left')
    my_palette = sns.color_palette("hls", len(descr_df[categ]))
    plt.figure(figsize=(10,10))
    ax =sns.scatterplot(data=df1,x=x_col, y=y_col, hue=descr, palette=my_palette)
    plt.title(plt_title, fontsize=20)
    plt.legend(fontsize='16', loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

# a function to produce a scatter plot
def plotly_scat(df,cat, titl):
    
    i=2
    import plotly.express
#     x = df.columns[-2]
#     y = df.columns[-1]
    df['x'] = df.Low_dim.apply(lambda x: x[2*i])
    df['y'] = df.Low_dim.apply(lambda x: x[2*i +1])
    x = df.x
    y = df.y
    
    if len(cat) == 1:
        # get a dataframe with descriptions for the category level we are investigating
        descr_df = get_category_description(cat)
        categ = descr_df.columns[0]
        descr = descr_df.columns[1]
        plot_df = df.merge(descr_df, on = categ,  how='left')
    else:
        plot_df = df1[[categ,"Descr_cleaned", "Code", "Low_dim","label"]].copy()
       # plot_df['Category'] = plot_df['Category_1'].astype(str)

        categ = cat

    hover = {
        "x": False,
        "y": False
        }
    true_cols = ['Descr_cleaned','label']
    for col in true_cols:
        hover[col] = True
    fig = plotly.express.scatter(
            plot_df, 
            x="x", 
            y="y", 
            color=categ,
            title=titl,
            hover_data=hover)
    
    return fig

# a function to produce a scatter plot
def scat(plot_df, true_cols):
    import plotly.express
    hover = {
        "Category":False,
        "x": False,
        "y": False
        }
    for col in true_cols:
        hover[col] = True
    fig = plotly.express.scatter(
            plot_df, 
            x="x", 
            y="y", 
            color="Category",
            hover_data=hover)
    return fig


def classifier(vector_col: pd.Series, target_col: pd.Series):
    
    logger.info("Now training random forest classifier")
    vecs = np.array(list(vector_col.values))
    target = np.array(list(target_col.values))
    
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100
                              ).fit(vecs, target)
    return clf


def cluster(vector_col: pd.Series):

    vecs = np.array(list(vector_col))

    labels1 = hdbscan.HDBSCAN().fit_predict(vecs)
    labels = pd.Series(data=labels1.tolist(), index=vector_col.index).astype(str)

    return labels


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
    

def show_group(df, search_col, start_str):
    l = len(start_str)
    df1 = df[df[search_col].str[:l] == start_str]
    #cols = ['Code', 'Level', 'Descr_old', 'Descr', 'Includes', 'Full_descr', 'Descr_cleaned','Full_descr_cleaned',  'label']
    cols = ['Code', 'Level',  'Descr_cleaned','Full_descr_cleaned',  'label']
    return df1[cols]
    
def search_col(df, search_col, st):
    st1 = st.lower()
    df1 = df[df[search_col].fillna('').str.contains(st1)]
    #cols = ['Code', 'Level', 'Descr_old', 'Descr', 'Includes', 'Full_descr', 'Descr_cleaned','Full_descr_cleaned',  'label']
    return df1


if __name__ == "__main__":

    hello_you()
    
