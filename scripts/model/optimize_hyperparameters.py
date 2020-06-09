
#####################
### Configuration
#####################

## Path to User-item Data
DATA_FILE = "./data/raw/user_item/2020-02-21_2020-02-28/comment_user_item_matrix.joblib"

## Data Parameters
MIN_SUPPORT = 25
MIN_HISTORY = 5

## Hyperparameters For Search
N_FACTORS = [10, 25, 50, 100, 250, 500, 1000]
REGULARIZATION = [1e-4,1e-3,1e-2,1e-1,1,10,100]

## Training Parameters
ITERATIONS = 25
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
import sys

## External Libaries
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## Local
from rrec.model.collaborative_filtering import CollaborativeFiltering
from rrec.util.logging import initialize_logger

LOGGER = initialize_logger()

#####################
### Helpers
#####################

## Evaluate Recommendation Performance
def score_model(model,
                user_item,
                user_item_rows,
                test_sample_size,
                evaluation_sample_size=5000):
    """

    """
    np.random.seed(RANDOM_STATE)
    eval_users = sorted(np.random.choice(user_item.shape[1], evaluation_sample_size, replace=False))
    model_rows = set(model._items)
    results = []
    for user in tqdm(eval_users, desc="Evaluation Test User", file=sys.stdout):
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

#####################
### Setup
#####################

LOGGER.info("Loading Data")

## Plotting and Model Directories
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

## Load Data
data = joblib.load(DATA_FILE)

## Parse
X = data["X"]
rows = data["rows"]
columns = data["columns"]

## Filter Users
user_mask = np.nonzero((X>0).sum(axis=0) >= MIN_HISTORY)[1]
X_masked = X[:, user_mask]
columns_masked = [columns[i] for i in user_mask]

#####################
### Train/Test Split
#####################

LOGGER.info("Splitting Dataset")

## User Sample Selection
np.random.seed(RANDOM_STATE)
train_users, test_users = train_test_split(list(range(X_masked.shape[1])),
                                           test_size=1-TRAIN_USER_SAMPLE_SIZE)
train_users = sorted(train_users)
test_users = sorted(test_users)
X_train = X_masked[:, train_users]
X_test = X_masked[:, test_users]

## Filter Items
train_mask = np.nonzero((X_train>0).sum(axis=1) >= MIN_SUPPORT)[0]
X_train_masked = X_train[train_mask]
rows_masked = [rows[i] for i in train_mask]

#####################
### Grid Search
#####################

LOGGER.info("Starting Grid Search")

## Cache of Results
all_results = []

## Cycle Through Model Parameters
for n in N_FACTORS:
    for reg in REGULARIZATION:
        LOGGER.info(f"Factors: {n}, Regulariztion: {reg}")
        ## Initialize Model
        cf = CollaborativeFiltering(factors=n,
                                    regularization=reg,
                                    iterations=ITERATIONS,
                                    num_threads=NUM_THREADS,
                                    random_state=RANDOM_STATE)
        ## Fit Model
        cf = cf.fit(X_train_masked,
                    rows=rows_masked,
                    columns=[columns_masked[i] for i in train_users])
        ## Get Results
        train_results = score_model(cf, X_train_masked, rows_masked, TEST_HISTORY_SAMPLE_SIZE, TEST_USER_SAMPLE_SIZE)
        test_results = score_model(cf, X_test, rows, TEST_HISTORY_SAMPLE_SIZE, TEST_USER_SAMPLE_SIZE)
        ## Append Information
        for df, user_group in zip([train_results, test_results], ["train","test"]):
            df["user_group"] = user_group
            df["n_factors"] = n
            df["regularization"] = reg
            all_results.append(df)

## Concatenate Results
all_results = pd.concat(all_results).reset_index(drop=True)

## Cache Results
all_results.to_csv("./data/processed/hyperparameter_search_results.csv")

#####################
### Analysis
#####################

LOGGER.info("Analyzing Results")

## Aggregate Performance
agg_cols = ["auc","recall_1","recall_5","recall_10","recall_25","recall_50"]
results_agg = pd.pivot_table(all_results,
                             index = ["user_group", "group", "regularization"],
                             columns = ["n_factors"],
                             values = agg_cols,
                             aggfunc = np.mean)

## Plot Aggregate Results (Heatmap)
for metric in agg_cols:
    fig, ax = plt.subplots(2, 2, figsize=(10,5.8))
    for u, ug in enumerate(["train","test"]):
        for g, gg in enumerate(["train","test"]):
            plot_data = results_agg.loc[ug, gg][metric]
            ax[u, g].imshow(plot_data.values,
                            cmap=plt.cm.Blues,
                            interpolation="nearest",
                            aspect="auto")
            ax[u, g].set_xticks(np.arange(plot_data.shape[1]))
            ax[u, g].set_xticklabels(plot_data.columns.tolist())
            ax[u, g].set_yticks(np.arange(plot_data.shape[0]))
            ax[u, g].set_yticklabels(plot_data.index.tolist())
            for x, xval in enumerate(plot_data.columns.tolist()):
                for y, yval in enumerate(plot_data.index.tolist()):
                    color = "white" if plot_data.values[y, x] > (np.max(plot_data.values) + np.min(plot_data.values)) / 2 else "black"
                    ax[u, g].text(x, y, "{:.3f}".format(plot_data.values[y, x]), color=color, ha="center", va="center") 
            ax[u, g].set_title(f"User Group: {ug.title()}, Split: {gg.title()}", loc="left", fontsize=8)
            ax[u, g].set_xlabel("# Factors")
            ax[u, g].set_ylabel("Regularization")
    fig.tight_layout()
    fig.suptitle(f"Metric: {metric}", fontweight="bold", x=0.15, y=.95)
    fig.subplots_adjust(top=.875)
    plt.savefig(f"{PLOT_DIR}grid_search_{metric}.png", dpi=200)
    plt.close()

LOGGER.info("Script Complete.")
