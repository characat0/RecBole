"""
ALS
################################################
Reference:
    Yifan Hu et al. "Collaborative Filtering for Implicit Feedback Datasets." in ICDM 2008.
"""

import torch
import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.utils import check_random_state

from recbole.utils import InputType, ModelType
from recbole.model.abstract_recommender import GeneralRecommender


class ALS(GeneralRecommender):
    r"""ALS is a matrix factorization model that minimizes the loss by using Alternating Least Squares.

    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(ALS, self).__init__(config, dataset)
        self.random_state = config['seed']
        self.embedding_size = config['embedding_size']
        self.alpha = config['alpha']
        self.regularization = config['reg_weight']
        self.cg_steps = config['cg_steps']
        # load parameters info
        self.reg_weight = config['reg_weight']
        self._training_loss = 0
        self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        self._fit_callback = lambda _, __, loss: self.__setattr__('_training_loss', loss)
        self.als = AlternatingLeastSquares(
            factors=self.embedding_size,
            alpha=self.alpha,
            regularization=self.regularization,
            iterations=1,
            random_state=self.random_state,
            calculate_training_loss=True,
        )
        self.als.cg_steps = self.cg_steps

    def calculate_loss(self, _):
        self.als.fit(self.interaction_matrix, show_progress=False, callback=self._fit_callback)
        return self._training_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_embedding = self.als.user_factors[user, :]
        item_embedding = self.als.item_factors[item, :]

        # We calculate the sum because item is repeated
        return torch.from_numpy(
            (user_embedding @ item_embedding)
            .sum(axis=1)
            .getA1()
        )

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        r = self.als.user_factors[user, :] @ self.als.item_factors.T
        return torch.from_numpy(r.flatten())
