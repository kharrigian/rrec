

## Database Paths
DB_PATH = "./data/db/"
USER_HISTORY_DB_PATH = f"{DB_PATH}user_history.db"
RECOMMENDATION_HISTORY_DB_PATH = f"{DB_PATH}recommendations.db"

## Model Path
MODEL_PATH = "./models/comments_20200221_20200228.cf"

## Temporal Boundaries
GLOBAL_START_DATE = "2018-01-01"

#################
### Imports
#################

## Standard Library
import os
import sys
import argparse
from datetime import datetime, timedelta

## External Libraries
import joblib
import numpy as np
import pandas as pd
import sqlite3 as sql
from tqdm import tqdm

## 
from rrec.acquire.reddit import RedditData
from rrec.model.reddit_recommender import RedditRecommender

#################
### Functions
#################

def parse_command_line():
    """

    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Collect location self-identification data")
    ## Generic Arguments
    parser.add_argument("database_path",
                        help="Path to Site Database")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Database Path Existence
    if not os.path.exists(args.database_path):
        raise FileNotFoundError(f"Could not find database path: {args.database_path}")
    return args

def retrieve_active_users(database_path):
    """

    """
    ## Read in database 
    ## TODO: Hook Up With Actual Database of Active Users
    active_users = ["FrenchLearnerJHU",
                    "TeachMeGermanJHU",
                    "HuskyKeith",
                    "SportsNotSprouts"]
    return active_users

def initialize_history_db():
    """

    """
    ## Initialize Connection
    user_history_con = sql.connect(USER_HISTORY_DB_PATH)
    ## Setup History Table
    init_command = """
    CREATE TABLE IF NOT EXISTS HISTORY (
    USER VARCHAR(200) NOT NULL,
    QUERY_START_DATE DATE NOT NULL,
    QUERY_END_DATE DATE NOT NULL,
    SUBREDDIT VARCHAR(200) NOT NULL,
    COMMENT_COUNT INTEGER NOT NULL,
    PRIMARY KEY (USER, QUERY_START_DATE, QUERY_END_DATE, SUBREDDIT));"""
    ## Run Initialization
    cursor = user_history_con.cursor()
    res = cursor.execute(init_command)
    ## Commit Changes and Close
    user_history_con.commit()
    user_history_con.close()

def initialize_recommendation_db():
    """

    """
    ## Initialize Connection
    rec_con = sql.connect(RECOMMENDATION_HISTORY_DB_PATH)
    ## Setup Recommendation Table
    init_command = """
    CREATE TABLE IF NOT EXISTS RECOMMENDATIONS (
        USER VARCHAR(200) NOT NULL,
        REC_DATE DATE NOT NULL,
        SUBREDDIT VARCHAR(200) NOT NULL,
        SCORE REAL NOT NULL,
        PRIMARY KEY (USER, SUBREDDIT, REC_DATE));"""
    ## Run Initialization
    cursor = rec_con.cursor()
    res = cursor.execute(init_command)
    ## Commit Changes and Close
    rec_con.commit()
    rec_con.close()

def update_user_histories(active_users):
    """

    """
    ## Get Today's Date
    today = datetime.now().date()
    ## Initialize Connection
    user_history_con = sql.connect(USER_HISTORY_DB_PATH)
    cursor = user_history_con.cursor()
    ## Get User Query Dates
    query_dates = {}
    max_date_command = """
    SELECT USER, QUERY_END_DATE
    FROM HISTORY
    WHERE USER='{}'
    ORDER BY QUERY_END_DATE
    LIMIT 1;"""
    for user in tqdm(active_users, total=len(active_users), desc="Query Periods", file=sys.stdout):
        res = cursor.execute(max_date_command.format(user))
        result = res.fetchall()
        if len(result) == 0:
            query_dates[user] = (GLOBAL_START_DATE, today.isoformat())
        else:
            if result[0][1] != today.isoformat():
                query_dates[user] = (result[0][1], today.isoformat())
    ## Initialize Reddit Wrapper
    reddit = RedditData()
    ## Query Comment History
    user_comment_histories = []
    for user, (start, stop) in tqdm(query_dates.items(), total=len(query_dates), file=sys.stdout, desc="User Histories"):
        df = reddit.retrieve_author_comments(user, start_date=start, end_date=stop)
        subreddit_counts = df.groupby(["author"])["subreddit"].value_counts().rename("COMMENT_COUNT").reset_index()
        subreddit_counts["QUERY_START_DATE"] = start
        subreddit_counts["QUERY_END_DATE"] = stop
        subreddit_counts.rename(columns={"author":"USER","subreddit":"SUBREDDIT"},inplace=True)
        user_comment_histories.append(subreddit_counts)
    ## Update Database
    if len(user_comment_histories) > 0:
        user_comment_histories = pd.concat(user_comment_histories).reset_index(drop=True)
        user_comment_histories.to_sql(name="HISTORY",
                                      con=user_history_con,
                                      if_exists="append",
                                      index=False,
                                    )
    ## Close Connection
    user_history_con.commit()
    user_history_con.close()

def update_subreddit_recommendations():
    """

    """
    ## Get Today's Date
    today = datetime.now()
    ## User Comment Histories
    user_history_con = sql.connect(USER_HISTORY_DB_PATH)
    cursor = user_history_con.cursor()
    res = cursor.execute("SELECT USER, SUBREDDIT, SUM(COMMENT_COUNT) FROM HISTORY GROUP BY SUBREDDIT, USER;")
    result = res.fetchall()
    ## Format Histories
    result = pd.DataFrame(result, columns=["USER","SUBREDDIT","COMMENT_COUNT"])
    users = result["USER"].unique()
    ## Load Model
    model = RedditRecommender(MODEL_PATH)
    ## Identify Most Popular Subreddits
    full_comment_distribution = pd.Series(index=model.cf._items,
                                          data=np.array(model.cf._item_user_matrix.sum(axis=1)).T[0])
    top_subreddits = full_comment_distribution.nlargest(100)
    top_subreddits = top_subreddits / top_subreddits.max()
    top_subreddits = pd.DataFrame(top_subreddits).rename(columns={0:"Recommendation Score"})
    top_subreddits.index.name="item"
    ## Get Top Recommendations Per User
    recommendations = []
    for user in tqdm(users, total=len(users), desc="Making Recommendations", file=sys.stdout):
        user_data = result.loc[result["USER"]==user].set_index("SUBREDDIT")["COMMENT_COUNT"].to_dict()
        user_recs = model.recommend(user_data, k_top=100, filter_liked=True)
        ## No-valid Recommendations (e.g. no user history matches)
        if user_recs.values.max() == 0:
            user_recs = top_subreddits.copy()
        ## Format
        user_recs = user_recs.reset_index().rename(columns={"item":"SUBREDDIT","Recommendation Score":"SCORE"})
        user_recs["USER"] = user
        user_recs["REC_DATE"] = today.isoformat()
        recommendations.append(user_recs)
    ## Concatenate Recommendations
    recommendations = pd.concat(recommendations).reset_index(drop=True)
    ## Upload Recommendations
    rec_db_con = sql.connect(RECOMMENDATION_HISTORY_DB_PATH)
    recommendations.to_sql("RECOMMENDATIONS", con=rec_db_con, if_exists="append",index=False)
    ## Commit and Close
    rec_db_con.commit()
    rec_db_con.close()


def main():
    """

    """
    ## Parse Command Line Arguments
    args = parse_command_line()
    ## Check/Created DB Directory
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    ## Initialize Databases if They Don't Exist
    _ = initialize_history_db()
    _ = initialize_recommendation_db()
    ## Get Active Website Users
    active_users = retrieve_active_users(args.database_path)
    ## Update User Histories
    _ = update_user_histories(active_users)
    ## Update Recommendations
    _ = update_subreddit_recommendations()

#######################
### Execute
#######################

if __name__ == "__main__":
    _ = main()