
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
MIN_HISTORY = 5

## Model Parameters
N_FACTORS = 250
ITERATIONS = 25
REGULARIZATION = 0.01
NUM_THREADS = 8
RANDOM_STATE = 42

## EVALUATION
TRAIN_USER_SAMPLE_SIZE = 0.8
TEST_USER_SAMPLE_SIZE = 5000
TEST_HISTORY_SAMPLE_SIZE = 0.5

## Outputs
PLOT_DIR = "./plots/"

#####################
### Imports
#####################

## Standard Library
import os

## External Libaries
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## Local
from rrec.acquire.reddit import RedditData
from rrec.model.collaborative_filtering import CollaborativeFiltering

#####################
### Helper Functions
#####################

## Evaluate Recommendation Performance
def score_model(model,
                user_item,
                user_item_rows,
                test_sample_size,
                evaluation_sample_size=5000):
    """

    """
    np.random.seed(42)
    eval_users = sorted(np.random.choice(user_item.shape[1], evaluation_sample_size, replace=False))
    model_rows = set(model._items)
    results = []
    for user in eval_users:
        user_history = user_item[:, user].T
        user_subreddits = np.nonzero(user_history)[1]
        if len(user_subreddits) < int(1 / test_sample_size):
            continue
        train_subs, test_subs = train_test_split(user_subreddits, test_size=test_sample_size)
        train_subs = dict(zip([user_item_rows[i] for i in train_subs], user_history[:,train_subs].toarray()[0]))
        test_subs = dict(zip([user_item_rows[i] for i in test_subs], user_history[:,test_subs].toarray()[0]))
        user_recs = model.recommend(train_subs, filter_liked=False, k_top=len(model_rows))
        for tset, subs in zip(["train","test"],[train_subs, test_subs]):
            fpr, tpr, thresh = metrics.roc_curve((user_recs["item"].isin(subs)).astype(int).tolist(),
                                                 user_recs["score"].tolist())
            hits = {"total_items":len(subs), "group":tset}
            hits["matched_items"] = len(set(subs) & model_rows)
            hits["fpr"] = fpr
            hits["tpr"] = tpr
            hits["auc"] = metrics.auc(fpr, tpr)
            for rec_thresh in [1, 5, 10, 25, 50]:
                rec_hits = len(set(user_recs.iloc[:rec_thresh]["item"]) & set(subs))
                hits[rec_thresh] = rec_hits
            results.append(hits)
    results = pd.DataFrame(results)
    for thresh in [1, 5, 10, 25, 50]:
        results[f"recall_{thresh}"] = results[thresh] / results["total_items"]
    return results

def recall_plot(results):
    """

    """
    fig, ax = plt.subplots(1, 5, figsize=(10, 5.8), sharey=True)
    for t, thresh in enumerate([1, 5, 10, 25, 50]):
        for g, group in enumerate(["train","test"]):
            plot_vals = results.loc[results["group"]==group][f"recall_{thresh}"]
            ax[t].hist(plot_vals, bins = np.linspace(0, 1, 11), density=True, label=group, alpha=.5)
        ax[t].set_title(f"Top {thresh}", loc="left")
        ax[t].legend(loc="upper left", frameon=True, fontsize=10)
        ax[t].set_xlabel("Recall", fontsize=10)
    ax[0].set_ylabel("Density", fontsize=10)
    fig.tight_layout()
    return fig, ax

def roc_auc_plot(results):
    """

    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 5.8), sharex=False, sharey=False)
    for g, group in enumerate(["train","test"]):
        plot_vals = results.loc[results["group"]==group]
        for r, row in plot_vals.iterrows():
            ax[0, g].plot(row["fpr"], row["tpr"], alpha=0.1, color="C0", linewidth=.5)
        ax[1, g].hist(plot_vals["auc"], bins=np.linspace(0, 1, 21), density=True)
    for i in range(2):
        ax[0, i].plot([0, 1], [0, 1], color="red", linestyle="--")
        ax[0, i].set_xlabel("False Positive Rate", fontsize=10)
        ax[0, i].set_ylabel("True Positive Rate", fontsize=10)
        ax[1, i].set_xlabel("AUC", fontsize=10)
        ax[1, i].set_ylabel("Density", fontsize=10)
    ax[0, 0].set_title("Within History", loc="left")
    ax[0, 1].set_title("Outside History", loc="left")
    fig.tight_layout()
    return fig, ax

#####################
### Setup
#####################

## Plotting and Model Directories
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

## Load Data
data = joblib.load(DATA_FILE)

## Parse
X = data["X"]
rows = data["rows"]
columns = data["columns"]

#####################
### Model Evaluation
#####################

## Filter Users
user_mask = np.nonzero((X>0).sum(axis=0) >= MIN_HISTORY)[1]
X_masked = X[:, user_mask]
columns_masked = [columns[i] for i in user_mask]

## User Sample Selection
np.random.seed(RANDOM_STATE)
train_users, test_users = train_test_split(list(range(X_masked.shape[1])), test_size=1-TRAIN_USER_SAMPLE_SIZE)
train_users = sorted(train_users)
test_users = sorted(test_users)
X_train = X_masked[:, train_users]
X_test = X_masked[:, test_users]

## Filter Items
train_mask = np.nonzero((X_train>0).sum(axis=1) >= MIN_SUPPORT)[0]
X_train_masked = X_train[train_mask]
rows_masked = [rows[i] for i in train_mask]

## Fit Model
cf = CollaborativeFiltering(factors=N_FACTORS,
                            regularization=REGULARIZATION,
                            iterations=ITERATIONS,
                            num_threads=NUM_THREADS,
                            random_state=RANDOM_STATE)
cf = cf.fit(X_train_masked,
            rows=rows_masked,
            columns=[columns_masked[i] for i in train_users])

## Get Results
train_results = score_model(cf, X_train_masked, rows_masked, TEST_HISTORY_SAMPLE_SIZE, TEST_USER_SAMPLE_SIZE)
test_results = score_model(cf, X_test, rows, TEST_HISTORY_SAMPLE_SIZE, TEST_USER_SAMPLE_SIZE)

## Plot Performance
for user_group, group_results in zip(["train","test"],[train_results, test_results]):
    ## Recall
    fig, ax = recall_plot(group_results)
    fig.savefig(f"{PLOT_DIR}{user_group}_cf_recall.png")
    plt.close()
    ## ROC/AUC
    fig, ax = roc_auc_plot(group_results)
    fig.savefig(f"{PLOT_DIR}{user_group}_cf_roc_auc.png")
    plt.close()

#####################
### Fit Full Model
#####################

## Filter Users
user_mask = np.nonzero((X>0).sum(axis=0) >= MIN_HISTORY)[1]
X_masked = X[:, user_mask]
columns_masked = [columns[i] for i in user_mask]

## Filter Subreddits
subreddit_mask = np.nonzero((X_masked>0).sum(axis=1) >= MIN_SUPPORT)[0]
X_masked = X_masked[subreddit_mask]
rows_masked = [rows[i] for i in subreddit_mask]

## Fit Model
cf = CollaborativeFiltering(factors=N_FACTORS,
                            regularization=REGULARIZATION,
                            iterations=ITERATIONS,
                            num_threads=NUM_THREADS,
                            random_state=RANDOM_STATE)
cf = cf.fit(X_masked,
            rows=rows_masked,
            columns=columns_masked)

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