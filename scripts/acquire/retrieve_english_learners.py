
##################
### Imports
##################

## Standard Library
import os
import sys

## External Library
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

## Local
from rrec.acquire.reddit import RedditData

##################
### Configuration
##################

## Time Frame
START_DATE = "2020-01-01"
END_DATE = "2020-04-01"

## Search Terms
SEARCH_TERMS = [
                "sorry for the bad english",
                "not my native",
                "still learning english"
                ]

STRICT_FILTER = False

##################
### Query Data
##################

## Initialize Reddit Data Wrapper
reddit = RedditData(False)

## Cycle Through Search Terms
search_results = []
for ST in tqdm(SEARCH_TERMS, total=len(SEARCH_TERMS), file=sys.stdout):
    st_results = reddit.search_for_comments(query=ST,
                                            subreddit=None,
                                            start_date=START_DATE,
                                            end_date=END_DATE)
    st_results["search_term"] = ST
    if STRICT_FILTER:
        st_results = st_results.loc[st_results["body"].str.lower().str.contains(ST)]
    search_results.append(st_results)

## Concatenate Data
search_results = pd.concat(search_results)
search_results = search_results.reset_index(drop=True)

## De-duplicate, Preserving Search Term Matches
search_term_map = search_results.groupby(["id"])["search_term"].unique()
search_term_map = search_term_map.to_dict()
search_results = search_results.drop_duplicates(subset=["id"]).copy()
search_results["search_term"] = search_results["id"].map(lambda i: search_term_map[i])

## Remove Bots/Spam
FILTER_USERS = ["AutoModerator","KeepingDankMemesDank"]
search_results = search_results.loc[~search_results["author"].isin(FILTER_USERS)]

## Reset Index
search_results = search_results.reset_index(drop=True)