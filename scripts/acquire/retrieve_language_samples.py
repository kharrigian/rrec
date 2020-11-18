
######################
### Configuration
######################

## Start and End Date for Querying Date
START_DATE = "2019-07-01"
END_DATE = "2020-03-01"

## Parameters
MIN_ACTIVITY = 20
SAMPLE_SIZE = 5000

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
CACHE_DIR = "./data/raw/language/"

## Logger
logger = initialize_logger()

######################
### Functions
######################
    
def parse_arguments():
    """
    Parse command-line to identify configuration filepath.
    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Collect location self-identification data")
    ## Generic Arguments
    parser.add_argument("--comments",
                        default=False,
                        action="store_true",
                        help="If included, retrieves comment-based user-item matrix")
    parser.add_argument("--submissions",
                        default=False,
                        action="store_true",
                        help="If included, retrieves submission-based user-item matrix")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not args.comments and not args.submissions:
        raise ValueError("No data collection flags were specified")
    return args

def create_cache_directories(args):
    """

    """
    main_cache_dir = f"{CACHE_DIR}{START_DATE}_{END_DATE}/"
    comment_cache_dir = None
    submission_cache_dir = None
    if args.comments:
        comment_cache_dir = f"{main_cache_dir}comments/"
    if args.submissions:
        submission_cache_dir = f"{main_cache_dir}submissions/"
    for dr in [main_cache_dir, comment_cache_dir, submission_cache_dir]:
        if dr is not None and not os.path.exists(dr):
            os.makedirs(dr)
    return main_cache_dir, comment_cache_dir, submission_cache_dir

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

def retrieve_language_samples(active_subreddits,
                              cache_dir,
                              sample_type="comment"):
    """

    """
    ## Subreddits Meeting Criteria
    subreddits = active_subreddits[active_subreddits >= MIN_ACTIVITY]
    ## Initialize Reddit
    reddit = RedditData(False)
    ## Retreve Samples for Each Subreddit
    for subreddit, sample in tqdm(subreddits.iteritems(),
                                  total=len(subreddits),
                                  file=sys.stdout):
        ## Sample File
        sample_file = f"{cache_dir}{subreddit}.json"
        if os.path.exists(sample_file):
            continue
        ## Query Based on Data Type
        if sample_type == "comment":
            endpoint = reddit.search_for_comments
        elif sample_type == "submission":
            endpoint = reddit.search_for_submissions
        else:
            raise ValueError("Expected sample_type to be either comment or submission")
        df = endpoint(None,
                      subreddit=subreddit,
                      start_date=START_DATE, 
                      end_date=END_DATE,
                      limit=SAMPLE_SIZE)
        ## Extract Text Samples
        text_samples = dict()
        if len(df) > 0 and sample_type == "comment":
            text_samples = dict(zip(df["id"].tolist(),
                                    df["body"].tolist()))
        elif len(df) > 0 and sample_type == "submission":
            text_samples = dict(zip(df["id"].tolist(),
                                    (df["title"] + " " + df["selftext"]).tolist()))
        ## Cache Samples
        with open(sample_file, "w") as the_file:
            json.dump(text_samples, the_file)

def main():
    """

    """
    ## Parse Args
    args = parse_arguments()
    ## Create Cache Directories
    main_cache_dir, comment_cache_dir, submission_cache_dir = create_cache_directories(args)
    ## Retrieve Active Subreddits
    if SUBREDDIT_LIST is not None:
        active_subreddits = pd.read_csv(SUBREDDIT_LIST, low_memory=True, index_col=0)["0"]
    else:
        subreddit_out_file = f"{main_cache_dir}active_subreddits.csv"
        active_subreddits = get_active_subreddits()
        active_subreddits.to_csv(subreddit_out_file)
    ## Retrieve Language Samples
    if args.comments:
        _ = retrieve_language_samples(active_subreddits,
                                      comment_cache_dir,
                                      sample_type="comment")
    if args.submissions:
        _ = retrieve_language_samples(active_subreddits,
                                      submission_cache_dir,
                                      sample_type="submission")
    logger.info("Script Complete")

######################
### Execution
######################
    
if __name__ == "__main__":
    _ = main()