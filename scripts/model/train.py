
#####################
### Configuration
#####################

## Path to User-item Data
DATA_FILE = "./data/raw/user_item/2020-02-21_2020-02-28/comment_user_item_matrix.joblib"

## Model Name
MODEL_DIR = "./models/"
MODEL_NAME = "comments_20200221_20200228.cf"

## Data Parameters
MIN_SUPPORT = 25
BM25_WEIGHTING = False

## Model Parameters
N_FACTORS = 100
ITERATIONS = 25
REGULARIZATION = 10
NUM_THREADS = 8

#####################
### Imports
#####################

## External Libaries
import joblib
import numpy as np
from implicit.nearest_neighbours import bm25_weight

## Local
from rrec.acquire.reddit import RedditData
from rrec.model.collaborative_filtering import CollaborativeFiltering

#####################
### Process
#####################

## Load Data
data = joblib.load(DATA_FILE)

## Parse
X = data["X"]
rows = data["rows"]
columns = data["columns"]

## Filter
mask = np.nonzero((X>0).sum(axis=1) >= MIN_SUPPORT)[0]
X_masked = X[mask]
rows_masked = [rows[i] for i in mask]

## Weight Using BM25
if BM25_WEIGHTING:
    X_masked = bm25_weight(X_masked).tocsr()

## Fit Model
cf = CollaborativeFiltering(factors=N_FACTORS,
                            regularization=REGULARIZATION,
                            iterations=ITERATIONS,
                            num_threads=NUM_THREADS)
cf = cf.fit(X_masked,
            rows=rows_masked,
            columns=columns)

#####################
### Testing
#####################

## Test Recommendations
reddit = RedditData()
keith = reddit.retrieve_author_comments("HuskyKeith")
keith_counts = keith["subreddit"].tolist()
keith_recs = cf.recommend(keith_counts, 20)

## Test Similarity
cf.get_similar_item("movies")

## Dump Model
cf.dump(f"{MODEL_DIR}{MODEL_NAME}")