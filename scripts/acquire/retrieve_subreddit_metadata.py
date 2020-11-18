
####################
### Imports
####################

## Start and End Date for Querying Date (Relevant if Querying New Subreddits)
START_DATE = "2019-07-01"
END_DATE = "2020-03-01"

## Path to Subreddits List (None if it doesn't exist already)
SUBREDDIT_LIST = "./data/raw/user_item/2020-02-21_2020-02-28/active_subreddits.csv"

######################
### Imports
######################

## Standard Library
import os
import sys
import json
import argparse
from glob import glob

## External Libraries
import pandas as pd
from tqdm import tqdm

## Local
from rrec.acquire.reddit import RedditData
from rrec.util.logging import initialize_logger

######################
### Globals
######################

## Cache Directory
CACHE_DIR = "./data/raw/metadata/"

## Logger
logger = initialize_logger()

######################
### Functions
######################

def get_active_subreddits(search_freq="5m"):
    """

    """
    ## Initialize reddit Class
    reddit = RedditData(False)
    ## Retrieve Active Subreddits
    logger.info("Retrieving Active Subreddits")
    active_subreddits = reddit.identify_active_subreddits(START_DATE,
                                                          END_DATE,
                                                          chunksize=search_freq)
    return active_subreddits

def main():
    """

    """
    ## Directory Setup
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    temp_dir = f"{CACHE_DIR}temp/"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    ## Retrieve Active Subreddits
    if SUBREDDIT_LIST is not None:
        active_subreddits = pd.read_csv(SUBREDDIT_LIST, low_memory=True, index_col=0)["0"]
    else:
        subreddit_out_file = f"{main_cache_dir}active_subreddits.csv"
        active_subreddits = get_active_subreddits()
        active_subreddits.to_csv(subreddit_out_file)
    ## Retrieve Subreddit Metadata
    reddit = RedditData(True)
    for subreddit in tqdm(active_subreddits.index.tolist(),
                          file=sys.stdout,
                          total=len(active_subreddits),
                          desc="Subreddit"):
        sub_temp_file = f"{temp_dir}{subreddit}.json"
        if os.path.exists(sub_temp_file):
            continue
        sub_metadata = reddit.retrieve_subreddit_metadata(subreddit)
        if sub_metadata is None:
            continue
        with open(sub_temp_file, "w") as the_file:
            json.dump(sub_metadata, the_file)
    ## Concatenate Data
    cached_files = glob(temp_dir + "*")
    metadata = pd.DataFrame([json.load(open(f,"r")) for f in cached_files])
    metadata = metadata.sort_values("subscribers",ascending=False).reset_index(drop=True)
    ## Cache
    metadata.to_csv(f"{CACHE_DIR}subreddit_metadata.csv")
    logger.info("Script Complete")

######################
### Execution
######################
    
if __name__ == "__main__":
    _ = main()