
## TODO: Readability for non-English text

###########################
### Configuration
###########################

## Language Data Directory
LANGUAGE_DIR = "./data/raw/language/2020-02-21_2020-02-28/comments/"

###########################
### Imports
###########################

## Standard Library
import os
import json
from glob import glob
from multiprocessing import Pool

## External Libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from langdetect import detect_langs
from langid import classify
import readability
from scipy.sparse import vstack
from sklearn.feature_extraction import DictVectorizer

## Local
from rrec.util.helpers import flatten

###########################
### Globals
###########################

reading_indices = ['Kincaid',
                   'ARI',
                   'Coleman-Liau',
                   'FleschReadingEase',
                   'GunningFogIndex',
                   'LIX',
                   'SMOGIndex',
                   'RIX',
                   'DaleChallIndex']

sentence_info = ['characters_per_word',
                 'syll_per_word',
                 'words_per_sentence',
                 'sentences_per_paragraph',
                 'type_token_ratio',
                 'characters',
                 'syllables',
                 'words',
                 'wordtypes',
                 'sentences',
                 'paragraphs',
                 'long_words',
                 'complex_words',
                 'complex_words_dc']

###########################
### 
###########################

def infer_language(text,
                   method="langid"):
    """

    """
    if text is None or len(text) == 0:
        return None
    try:
        if method == "langid":
            lang, score = classify(text)
        elif method == "langdetect":
            l = detect_langs(text)
            lang = l.lang
            score = l.prob
        return lang
    except:
        return None

def infer_readability(text):
    """

    """
    if text is None or len(text) == 0:
        return None
    try:
        measures = readability.getmeasures(text)
    except:
        return None
    return measures

def summarize_language_sample(filename):
    """

    """
    ## Load Data
    with open(filename, "r") as the_file:
        language_data = json.load(the_file)
    subreddit = os.path.basename(filename)[:-5]
    language_data = pd.DataFrame({"text":language_data})
    ## Infer Language
    language_data["inferred_language"] = language_data["text"].map(infer_language)
    ## Readability Metrics
    language_data["readability"] = language_data["text"].map(infer_readability)
    for rl in reading_indices:
        language_data[rl] = language_data["readability"].map(lambda i: i["readability grades"][rl] if i is not None else np.nan)
    for si in sentence_info:
        language_data[si.replace(" ","_")] = language_data["readability"].map(lambda i: i["sentence info"][si] if i is not None else np.nan)
    language_data = language_data.drop("readability", axis=1)
    ## Averages
    averages = language_data.drop(["text","inferred_language"],axis=1).mean().to_dict()
    averages["language_distribution"] = language_data["inferred_language"].value_counts().to_dict()
    return subreddit, averages

def create_dict_vectorizer(vocab):
    """

    """
    ngram_to_idx = dict((n, i) for i, n in enumerate(sorted(vocab)))
    _count2vec = DictVectorizer(separator=":")
    _count2vec.vocabulary_ = ngram_to_idx.copy()
    rev_dict = dict((y, x) for x, y in ngram_to_idx.items())
    _count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
    return _count2vec

def process_language_samples():
    """
    """
    ## Identify Language Samples
    data_files = glob(f"{LANGUAGE_DIR}*.json")
    ## Process Samples
    language_summary = dict()
    for filename in tqdm(data_files):
        subreddit, stats = summarize_language_sample(filename)
        language_summary[subreddit] = stats
    ## Format
    language_summary = pd.DataFrame(language_summary).T
    ## Language Distribution
    unique_languages = sorted(set(flatten(language_summary["language_distribution"])))
    language_vectorizer = create_dict_vectorizer(unique_languages)
    language_X = vstack(language_summary["language_distribution"].map(language_vectorizer.transform).tolist()).toarray()
    language_dist = pd.DataFrame(language_X,
                                 columns=unique_languages, 
                                 index=language_summary.index.tolist())


