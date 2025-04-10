"""
ALS
################################################
Reference:
    Yifan Hu et al. "Collaborative Filtering for Implicit Feedback Datasets." in ICDM 2008.
"""

import torch
import numpy as np
from implicit.als import AlternatingLeastSquares

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
        self.cg_steps = config['cg_steps'] or 1
        # load parameters info
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self._training_loss = 0
        self.interaction_matrix = dataset.inter_matrix(form='csr', value_field=config['RATING_FIELD']).astype(np.float32)
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
        self.i_factor = None

        self.other_parameter_name = ['als', 'random_state', 'embedding_size', 'alpha', 'regularization', 'cg_steps']

    def calculate_loss(self, _=None):
        return self._training_loss

    def train_epoch(self):
        self.als.fit(self.interaction_matrix, show_progress=False, callback=self._fit_callback)
        return self.calculate_loss()

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu()
        item = interaction[self.ITEM_ID].cpu()

        user_embedding = self.als.user_factors[user, :]
        item_embedding = self.als.item_factors[item, :].T

        # We calculate the sum because item is repeated
        return torch.from_numpy(
            (user_embedding @ item_embedding)
            .sum(axis=1)
        ).to(self.device)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu()
        u_factor = torch.from_numpy(self.als.user_factors[user, :]).to(self.device)
        if self.i_factor is None or (self.i_factor.size(0) != u_factor.size(1)):
            self.i_factor = torch.from_numpy(self.als.item_factors.T).to(self.device)
        r = u_factor @  self.i_factor
        return r.view(-1)
