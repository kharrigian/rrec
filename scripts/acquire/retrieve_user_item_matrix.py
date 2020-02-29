
######################
### Configuration
######################

## Start and End Date for Querying Date
START_DATE = "2020-02-21"
END_DATE = "2020-02-28"

## Overwrite Existing Data
OVERWRITE_SEARCH = False

######################
### Imports
######################

## Standard Library
import os
import json
import gzip
import argparse
from glob import glob

## External
import joblib
from scipy.sparse import vstack
from sklearn.feature_extraction import DictVectorizer

## Local
from rrec.acquire.reddit import RedditData
from rrec.util.logging import initialize_logger

######################
### Globals
######################

## Cache Directory
CACHE_DIR = "./data/raw/user_item/"

## Logger
logger = initialize_logger()

######################
### Helper Functions
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

def create_dict_vectorizer(vocab):
    """

    """
    ngram_to_idx = dict((n, i) for i, n in enumerate(sorted(vocab)))
    _count2vec = DictVectorizer(separator=":")
    _count2vec.vocabulary_ = ngram_to_idx.copy()
    rev_dict = dict((y, x) for x, y in ngram_to_idx.items())
    _count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
    return _count2vec

def get_active_subreddits(search_freq=5):
    """

    """
    ## Initialize reddit Class
    reddit = RedditData(False)
    ## Retrieve Active Subreddits
    logger.info("Retrieving Active Subreddits")
    active_subreddits = reddit.identify_active_subreddits(START_DATE,
                                                          END_DATE,
                                                          search_freq=search_freq)
    return active_subreddits

def get_user_item_matrix(active_subreddits,
                         cache_dir,
                         history_type="comment"):
    """

    """
    ## Initialize reddit Class
    reddit = RedditData(False)
    ## Collect Post Histories
    subreddits = active_subreddits.index.tolist()
    total_s = len(subreddits)
    for s, subreddit in enumerate(subreddits):
        logger.info("Collecting Subreddit {}/{}: {}".format(s+1, total_s, subreddit))
        subreddit_file = f"{cache_dir}{subreddit}.tar.gz"
        if os.path.exists(subreddit_file):
            continue
        author_post_history = reddit.retrieve_subreddit_user_history(subreddit,
                                                                     start_date=START_DATE,
                                                                     end_date=END_DATE,
                                                                     history_type=history_type,
                                                                     docs_per_chunk=5000)
        with gzip.open(subreddit_file, "wt") as the_file:
            if author_post_history is not None:
                json.dump(author_post_history.to_dict(), the_file)
            else:
                json.dump({}, the_file)
    ## Identify Unique Redditors
    redditors = set()
    history_files = sorted(glob(cache_dir + "*.tar.gz"))
    for h in history_files:
        with gzip.open(h, "r") as the_file:
            h_data = json.load(the_file)
            redditors.update(set(h_data))
    redditors = sorted(redditors)
    ## Create User-Item Matrix
    dvec = create_dict_vectorizer(redditors)
    X = []
    for h in history_files:
        with gzip.open(h, "r") as the_file:
            h_data = json.load(the_file)
            x = dvec.transform(h_data)
            X.append(x)
    ## Format
    X = vstack(X)
    rows = list(map(lambda h: os.path.basename(h)[:-7], history_files))
    columns = redditors
    return X, rows, columns
    
def create_cache_directories():
    """

    """
    main_cache_dir = f"{CACHE_DIR}{START_DATE}_{END_DATE}/"
    comment_cache_dir = f"{main_cache_dir}comments/"
    submission_cache_dir = f"{main_cache_dir}submissions/"
    for dr in [main_cache_dir, comment_cache_dir, submission_cache_dir]:
        if not os.path.exists(dr):
            os.makedirs(dr)
    return main_cache_dir, comment_cache_dir, submission_cache_dir

def main():
    """

    """
    ## Parse Args
    args = parse_arguments()
    ## Create Cache Directories
    main_cache_dir, comment_cache_dir, submission_cache_dir = create_cache_directories()
    ## Retrieve Active Subreddits
    active_subreddits_file = f"{main_cache_dir}active_subreddits.csv"
    if os.path.exists(active_subreddits_file) and not OVERWRITE_SEARCH:
        active_subreddits = pd.read_csv(active_subreddits_file, low_memory=True)
    else:
        active_subreddits = get_active_subreddits()
        active_subreddits.to_csv(active_subreddits_file)
    ## Retrieve Data
    if args.comments:
        X, subreddits, redditors = get_user_item_matrix(active_subreddits,
                                                        cache_dir=comment_cache_dir,
                                                        history_type="comment")
        joblib.dump({"X":X,"rows":subreddits,"columns":redditors},
                    f"{main_cache_dir}comment_user_item_matrix.joblib",
                    compress=3)
    if args.submissions:
        X, subreddits, redditors = get_user_item_matrix(active_subreddits,
                                                        cache_dir=submission_cache_dir,
                                                        history_type="submission")
        joblib.dump({"X":X,"rows":subreddits,"columns":redditors},
                    f"{main_cache_dir}submission_user_item_matrix.joblib",
                    compress=3)

######################
### Execute
######################

if __name__ == "__main__":
    _ = main()