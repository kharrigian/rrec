
#####################
### Imports
#####################

## Standard Libary
from collections import Counter

## External Libaries
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

#####################
### Class
#####################

class CollaborativeFiltering(object):
    
    """

    """

    def __init__(self,
                 factors=100,
                 regularization=0.01,
                 iterations=15,
                 calculate_training_loss=True,
                 num_threads=0,
                 random_state=42,
                 **kwargs):
        """

        """
        ## Initialize
        self._random_state = random_state
        self.model = AlternatingLeastSquares(factors=factors,
                                             regularization=regularization,
                                             iterations=iterations,
                                             calculate_training_loss=calculate_training_loss,
                                             num_threads=num_threads,
                                             **kwargs)
    
    def __repr__(self):
        """

        """
        desc = f"CollaborativeFiltering(factors={self.model.factors}, regularization={self.model.regularization})"
        return desc
    
    def fit(self,
            X,
            rows=None,
            columns=None):
        """
        Args:
            X (csr sparse matrix): Rows are Items, Columns are Users
            rows (list): Identifiers for the rows
            columns (list): Identifiers for the columns
        """
        ## Fix Random Seed
        np.random.seed(self._random_state)
        ## Cache Data
        self._item_user_matrix = X
        ## Indices
        if rows is None:
            rows = list(range(X.shape[0]))
        if columns is None:
            columns = list(range(X.shape[1]))
        self._item_map = dict((row, r) for r, row in enumerate(rows))
        self._items = rows
        self._user_map = dict((col, c) for c, col in enumerate(columns))
        self._users = columns
        ## Fit
        self.model.fit(X, show_progress=self.model.calculate_training_loss)
        return self
    
    def get_similar_item(self,
                         item,
                         k_top=10):
        """
        Find similar items to a given item using cosine similarity

        Args:
            item (any): One of the rows in the training data
            k_top (int): Number of top similar items to return
        """
        if item not in self._item_map:
            raise KeyError("Item does not exist")
        ## Compute Cosine Similarity
        item_f = self.model.item_factors[self._item_map[item]]
        item_factors = self.model.item_factors
        sim = item_factors.dot(item_f) / (self.model.item_norms * self.model.item_norms[self._item_map[item]])
        ## Select Top-k
        best = np.argpartition(sim, -k_top)[-k_top:]
        sim = sorted(zip(best, sim[best]), key=lambda x: -x[1])
        ## Replace Indices with Names
        sim_items = list(map(lambda i: [self._items[i[0]],i[1]], sim))
        sim_items = pd.DataFrame(sim_items, columns = ["item","similarity"])
        return sim_items

    def _user_linear_equation(self,
                              user_vector):
        """

        """
        A = self.model.YtY + self.model.regularization * np.eye(self.model.factors)
        b = np.zeros(self.model.factors)
        nonzeros = np.nonzero(user_vector)[0]
        for i, confidence in zip(nonzeros, user_vector[nonzeros]):
            factor = self.model.item_factors[i]
            if confidence > 0:
                b += confidence * factor
            else:
                confidence *= -1
            A += (confidence - 1) * np.outer(factor, factor)
        return A, b

    def recommend(self,
                  user_history,
                  filter_liked=False,
                  filter_items=[],
                  k_top=10):
        """
        Args:
            user_history (dict or list of raw items):
            k_top (int): Number of items to recommend
        """
        ## User History
        user_history = Counter(user_history)
        ## Compute User Factor
        user_vector = np.zeros(self.model.item_factors.shape[0])
        for item, count in user_history.items():
            if item not in self._item_map:
                continue
            user_vector[self._item_map[item]] = count
        ## Compute Score
        scores = self.model.recommend(userid=0,
                                      user_items=csr_matrix(user_vector),
                                      N=k_top,
                                      filter_already_liked_items=filter_liked,
                                      filter_items=list(map(lambda f: self._item_map[f], filter_items)),
                                      recalculate_user=True)
        ## Replace Indices with Names
        rec_items = list(map(lambda i: [self._items[i[0]],i[1]], scores))
        rec_items = pd.DataFrame(rec_items, columns = ["item","score"])
        return rec_items
    
    def dump(self,
             model_file,
             compress=3):
        """

        """
        _ = joblib.dump(self, model_file, compress=compress)
        
