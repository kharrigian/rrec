
## TODO: Readability for non-English text

###########################
### Configuration
###########################

## Language Data Directory
LANGUAGE_DIR = "./data/raw/language/2019-07-01_2020-03-01/comments/"

## Processed Data
PROCESSED_DATA_DIR = "./data/processed/"

## Multiprocessing
NUM_PROCESSES = 8

###########################
### Imports
###########################

## Standard Library
import os
import sys
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
import matplotlib.pyplot as plt

## Local
from rrec.util.helpers import flatten
from rrec.model.reddit_recommender import get_language_name

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
### Analysis Functions
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

def get_statistics(language_data,
                   min_threshold=5):
    """

    """
    ## Focus on Comments of Reasonable Length
    language_data = language_data.loc[language_data["words"] >= min_threshold].copy()
    if len(language_data) == 0:
        return {}
    ## Get Totals
    totals = language_data.drop(["text","inferred_language"],axis=1).sum()
    ## Get Statistics
    statistics = {
                "language_distribution":language_data["inferred_language"].value_counts().to_dict(),
                "avg_num_sent":language_data["sentences"].median(),
                "avg_num_words":language_data["words"].median(),
                "complex_word_ratio":totals["complex_words"] / totals["words"],
                "token_type_ratio":language_data["type_token_ratio"].median(),
                "support":len(language_data),
    }
    statistics.update(language_data[reading_indices].median().to_dict())
    return statistics

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
    ## Statistics
    language_stats = get_statistics(language_data)
    return subreddit, language_stats

def create_dict_vectorizer(vocab):
    """

    """
    ngram_to_idx = dict((n, i) for i, n in enumerate(sorted(vocab)))
    _count2vec = DictVectorizer(separator=":")
    _count2vec.vocabulary_ = ngram_to_idx.copy()
    rev_dict = dict((y, x) for x, y in ngram_to_idx.items())
    _count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
    return _count2vec

def get_language_distribution(language_summary):
    """

    """
    unique_languages = sorted(set(flatten(language_summary["language_distribution"])))
    language_vectorizer = create_dict_vectorizer(unique_languages)
    language_X = vstack(language_summary["language_distribution"].map(language_vectorizer.transform).tolist()).toarray()
    language_dist = pd.DataFrame(language_X,
                                 columns=unique_languages, 
                                 index=language_summary.index.tolist())
    language_dist = language_dist.loc[language_dist.sum(axis=1) > 0].copy()
    language_conc = language_dist.apply(lambda row: row / sum(row), axis = 1)
    return language_dist, language_conc

def _plot_overall_language_distribution(language_dist):
    """

    """
    ## Count
    language_counts = language_dist.sum(axis=0).sort_values(ascending=False)
    ## Re-label
    language_counts.index = language_counts.index.map(get_language_name)
    ## Organize Data of Top-k Languages
    highlights = language_counts.head(20).append(language_counts.tail(5))
    highlights = highlights[::-1]
    ind = list(range(26))
    ind_names = highlights.index[:5].tolist() + ["..."] + highlights.index[5:].tolist()
    values = highlights.tolist()[:5] + [0] + highlights.tolist()[5:]
    ## Create Figure
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.barh(ind,
            values,
            color="C0",
            alpha=.75)
    ax.set_yticks(ind)
    ax.set_yticklabels(ind_names)
    ax.set_xscale("symlog")
    ax.set_xlabel("Number of Comments in Sample")
    ax.set_ylim(-1, 26)
    fig.tight_layout()
    plt.savefig("./plots/language_distribution.png")
    plt.close()

def _plot_top_concentrations(language_dist,
                             language_conc,
                             min_support=100):
    """

    """
    ## Filter
    language_conc = language_conc.loc[language_dist.sum(axis=1)>=min_support]
    ## Create Plot Directory
    conc_dir = "./plots/concentration_by_language/"
    if not os.path.exists(conc_dir):
        os.makedirs(conc_dir)
    ## Plot Top for Each Language
    for language in language_conc.columns:
        language_top = language_conc[language].sort_values(ascending=True).tail(20) * 100
        language_name = get_language_name(language)
        fig, ax = plt.subplots(figsize=(10,5.8))
        language_top.plot.barh(ax=ax, color = "C0", alpha=.75)
        ax.set_xlabel("Percentage of Subreddit Comments")
        ax.set_title(f"Subreddits With Highest Concentration of {language_name} ({language})", loc="left")
        fig.tight_layout()
        fig.savefig(f"{conc_dir}{language}.png")
        plt.close()

def create_summary_plots(language_dist,
                         language_conc,
                         language_summary):
    """

    """
    _ = _plot_overall_language_distribution(language_dist)
    _ = _plot_top_concentrations(language_dist, language_conc, 100)

def main():
    """

    """
    ## Identify Language Samples
    data_files = glob(f"{LANGUAGE_DIR}*.json")
    ## Process Language Samples
    mp = Pool(NUM_PROCESSES)
    language_summary = dict(tqdm(mp.imap(summarize_language_sample, data_files),
                                 total=len(data_files),
                                 desc="File Count",
                                 file=sys.stdout))
    mp.close()
    ## Format
    language_summary = pd.DataFrame(language_summary).T
    language_summary = language_summary.dropna()
    ## Language Distribution
    language_dist, language_conc = get_language_distribution(language_summary)
    ## Cache Language Summary
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    language_summary.drop(["language_distribution"],axis=1).to_csv(f"{PROCESSED_DATA_DIR}subreddit_language_attributes.csv")
    language_dist.astype(int).to_csv(f"{PROCESSED_DATA_DIR}subreddit_language_distribution.csv")
    ## Create Summary Plots
    _ = create_summary_plots(language_dist, language_conc, language_summary)

###########################
### Execution
###########################

if __name__ == "__main__":
    _ = main()