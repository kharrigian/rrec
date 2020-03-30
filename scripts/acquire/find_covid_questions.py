
## Output Paths
OUTDIR = "./data/raw/covid/question_answer/"
OUTFILE = f"{OUTDIR}submission_search.csv"
COMMENT_OUTDIR = f"{OUTDIR}comments/"

##################
### Imports
##################

## Standard
import os
import sys
import json
import gzip

## External
from tqdm import tqdm
import pandas as pd

## Local
from rrec.acquire.reddit import RedditData
from rrec.util.logging import initialize_logger

LOGGER = initialize_logger()

##################
### Configuration
##################

## Time Range
START_DATE = "2019-11-01"
END_DATE = "2020-03-03"

## Create Search Combinations
SEARCH_TERMS = ["coronavirus",
                "covid",
                "flu",
                "social distance",
                "quarantine",
                "contagion",
                "CDC",
                "physician",
                "doctor",
                "pandemic",
                "nurse",
                "china",
                "bioweapon",
                "invisible enemy"
                ]
SUBREDDITS = ["AMA",
              "IAmA",
              "casualiama",
              "AskMeAnythingIAnswer"]

## Standalone Queries
STANDALONE_QUERIES = [
                ("coronavirus ama", None),
                ("coronavirus ask", None),
                ("coronavirus", "askdocs"),
                ("covid", "askdocs")
]

## Subreddits to Ignore
SUBREDDITS_TO_KEEP = [
    "AMA",
    "AskDocs",
    "IAmA",
    "casualiama",
    "Coronavirus",
    "AskMeAnythingIAnswer",
    "China_Flu",
    "cvnews",
    "China_Flu_Uncensored",
    "COVID19",
]

##################
### Functions
##################

def create_dir(directory):
    """

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
def main():
    """

    """
    ## Create Output Directories
    _ = create_dir(OUTDIR)
    _ = create_dir(COMMENT_OUTDIR)
    ## Initialize Reddit
    reddit = RedditData(False)
    ## Cycle Through General Search Terms
    search_results = []
    for subreddit in tqdm(SUBREDDITS, position=1, leave=False, desc="Subreddit", file=sys.stdout):
        for st in tqdm(SEARCH_TERMS, position=2, leave=False, desc="Search Terms", file=sys.stdout):
            search_res = reddit.search_for_submissions(query=st,
                                                       subreddit=subreddit,
                                                       start_date=START_DATE,
                                                       end_date=END_DATE,
                                                       )
            if search_res is not None and len(search_res) > 0:
                search_res["search_term"] = st
                search_results.append(search_res)
    ## Cycle Through Standlaone Search Terms
    for query, subreddit in tqdm(STANDALONE_QUERIES, file=sys.stdout):
        search_res = reddit.search_for_submissions(query=query,
                                                   subreddit=subreddit,
                                                   start_date=START_DATE,
                                                   end_date=END_DATE)
        if search_res is not None and len(search_res) > 0:
            search_res["search_term"] = st
            search_results.append(search_res)
    ## Concatenate Results
    search_results = pd.concat(search_results).reset_index(drop=True)
    ## Filter Out Some Subreddits
    search_results = search_results.loc[search_results.subreddit.isin(SUBREDDITS_TO_KEEP)]
    ## Search Term Map
    search_term_map = search_results.groupby(["id"])["search_term"].unique().to_dict()
    ## Drop Duplicates
    search_results = search_results.drop_duplicates(subset=["id"]).reset_index(drop=True)
    search_results["search_term"] = search_results["id"].map(lambda i: ", ".join(search_term_map[i]))
    ## Output Search Results
    search_results.to_csv(OUTFILE, index=False)
    ## Retrieve Comments
    urls = search_results.loc[search_results["num_comments"]>0]["full_link"].tolist()
    failed_retrievals = []
    for u in tqdm(urls, total=len(urls), file=sys.stdout):
        u_res = reddit.retrieve_submission_comments(u)
        if u_res is not None and len(u_res) > 0:
            submission_json = []
            for r, row in u_res.iterrows():
                submission_json.append(json.loads(row.to_json()))
            link_id = submission_json[0]["link_id"] 
            with gzip.open(f"{COMMENT_OUTDIR}{link_id}.json.gz","wt") as the_file:
                json.dump(submission_json, the_file)
        else:
            failed_retrievals.append(u)
    ## Failures
    LOGGER.info("Failed to Collect The Following URLS:")
    LOGGER.info(failed_retrievals)
    ## Complete
    LOGGER.info("\n\nScript Complete.")

##################
### Execute
##################

if __name__ == "__main__":
    main()