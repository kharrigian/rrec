
##########################
### Imports
##########################

## Standard Library
import os

## External Libraries
import joblib
import numpy as np
import pandas as pd
from langcodes import Language

## Local
from ..util.logging import initialize_logger

##########################
### Helpers
##########################

## Logging
logger = initialize_logger()

## Language Names
def get_language_name(chars):
    """

    """
    name = Language.get(chars).language_name()
    return name

##########################
### Class
##########################

class RedditRecommender(object):
    
    """

    """

    def __init__(self,
                 collaborative_filtering_path,
                 language_distribution_path=None,
                 min_support=10):
        """

        """
        ## Class Attributes
        self._cf_path = collaborative_filtering_path
        self._min_support = min_support
        ## Initialize Class
        self._initialize_collaborative_filtering(collaborative_filtering_path)
        self._initialize_language_distribution(language_distribution_path)
    
    def __repr__(self):
        """

        """
        desc = f"RedditRecommender(name={self._cf_path})"
        return desc 
    
    def _initialize_collaborative_filtering(self,
                                            path):
        """

        """
        logger.info("Loading Collaborative Filtering Model")
        if not os.path.exists(path):
            raise FileNotFoundError("Collaborative Filtering model not found")
        self.cf = joblib.load(path)
    
    def _initialize_language_distribution(self,
                                          path):
        """

        """
        logger.info("Loading Subreddit Language Details")
        if path is None:
            self._language_dist = None
            return
        if not os.path.exists(path):
            raise FileNotFoundError("Language Distribution not found")
        self._language_dist = pd.read_csv(path, index_col=0)
        ## Update Columns
        self._language_dist.rename(columns=dict((x, get_language_name(x)) for x in self._language_dist.columns),
                                   inplace=True)
        ## Filtering
        self._language_dist = self._language_dist.loc[self._language_dist.sum(axis=1) >= self._min_support].copy()
        ## Concentration
        self._language_conc = self._language_dist.apply(lambda row: row / sum(row), axis=1)
    
    def show_available_languages(self):
        """

        """
        languages = self._language_dist.columns.tolist()
        return languages

    def get_top_concentrations(self,
                               language,
                               k_top=20):
        """

        """
        top_concentrations = self._language_conc[language].sort_values(ascending=False).head(k_top)
        return top_concentrations
        
    def recommend(self,
                  user_history,
                  languages=[],
                  k_top=100,
                  **kwargs
                  ):
        """

        """
        ## Get Recommendations Based on Posting History
        cf_rec = self.cf.recommend(user_history, k_top=len(self.cf._items), **kwargs)
        cf_rec.set_index("item", inplace=True)
        ## Merge Language Distribution
        for language in languages:
            cf_rec[language] = self._language_conc[language]
        cf_rec = cf_rec.fillna(0)
        ## Get K Top
        cf_rec = cf_rec.head(k_top)
        cf_rec = cf_rec.rename(columns = {"score":"Recommendation Score"})
        return cf_rec