
## Output Directory for Data
OUTDIR = "./data/raw/covid/"

####################
### Imports
####################

## Standard Libary
import os
import sys
import json
import gzip

## External
from tqdm import tqdm

## Local
from rrec.acquire.reddit import RedditData
from rrec.util.logging import initialize_logger

LOGGER = initialize_logger()

####################
### Configuration
####################

## Time Span
START_DATE = "2020-02-01"
END_DATE = "2020-03-20"

## Subreddits
subreddits = [
              "COVID19"
             ]

## Submissions
submissions = [
              "https://www.reddit.com/r/Coronavirus/comments/fksnbf/im_bill_gates_cochair_of_the_bill_melinda_gates/"
              ]

####################
### Functions
####################

def create_dir(directory):
    """

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
def main():
    """

    """
    ## Initialize Reddit
    reddit = RedditData(True)
    ## Create Output Directory
    _ = create_dir(OUTDIR)
    ## Submissions
    LOGGER.info("Pulling Standalone Submission Comments")
    SUBMISSIONS_OUTDIR = f"{OUTDIR}submissions/"
    _ = create_dir(SUBMISSIONS_OUTDIR)
    for surl in tqdm(submissions, file=sys.stdout):
        submission_df = reddit.retrieve_submission_comments(surl)
        submission_json = []
        for r, row in submission_df.iterrows():
            submission_json.append(json.loads(row.to_json()))
        link_id = submission_json[0]["link_id"]
        soutfile = f"{SUBMISSIONS_OUTDIR}{link_id}.json.gz"
        with gzip.open(soutfile, "wt") as the_file:
            json.dump(submission_json, the_file)
    ## Subreddits
    LOGGER.info("Pulling Subreddit Metadata, Submissions, and Comments")
    for subreddit in subreddits:
        ## Create Output Directory
        SUBREDDIT_OUTDIR = f"{OUTDIR}subreddits/{subreddit}/"
        _ = create_dir(SUBREDDIT_OUTDIR)
        ## Pull Metadata
        LOGGER.info(f"Pulling Metadata for r/{subreddit}")
        subreddit_meta = reddit.retrieve_subreddit_metadata(subreddit)
        with gzip.open(f"{SUBREDDIT_OUTDIR}metadata.json.gz","wt") as the_file:
            json.dump(subreddit_meta, the_file)
        ## Identify Submission Data
        LOGGER.info(f"Pulling Submissions for r/{subreddit}")
        subreddit_submissions = reddit.retrieve_subreddit_submissions(subreddit,
                                                                      start_date=START_DATE,
                                                                      end_date=END_DATE)
        submission_json = []
        for r, row in subreddit_submissions.iterrows():
            submission_json.append(json.loads(row.to_json()))
        with gzip.open(f"{SUBREDDIT_OUTDIR}submissions.json.gz","wt") as the_file:
            json.dump(submission_json, the_file)
        ## Pull Comments
        LOGGER.info(f"Pulling Submissions for r/{subreddit}")
        SUBREDDIT_COMMENTS_DIR = f"{SUBREDDIT_OUTDIR}comments/"
        _ = create_dir(SUBREDDIT_COMMENTS_DIR)
        urls = subreddit_submissions.loc[subreddit_submissions["num_comments"] > 0]["id"].tolist()
        for url in tqdm(urls, file=sys.stdout):
            url_df = reddit.retrieve_submission_comments(url)
            if url_df is None or len(url_df) == 0:
                continue
            url_json = []
            for r, row in url_df.iterrows():
                url_json.append(json.loads(row.to_json()))
            link_id = url_json[0]["link_id"]
            url_outfile = f"{SUBREDDIT_COMMENTS_DIR}{link_id}.json.gz"
            with gzip.open(url_outfile,"wt") as the_file:
                json.dump(url_json, the_file)

####################
### Execute
####################

if __name__ == "__main__":
    main()