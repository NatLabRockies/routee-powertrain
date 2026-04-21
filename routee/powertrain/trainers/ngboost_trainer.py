import numpy as np
import pandas as pd

from ngboost import NGBRegressor
from ngboost.distns import Normal


from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.estimators.estimator_interface import Estimator
from routee.powertrain.estimators.ngboost_estimator import NGBoostEstimator
from routee.powertrain.trainers.trainer import Trainer


class NGBoostTrainer(Trainer):
    architecture_tag: str = "ngboost"

    def __init__(
        self,
        n_estimators: int = 100,
        dist=Normal,
        verbose: bool = True,
        verbose_eval: int = 20,
        learning_rate: float = 0.01,
        random_state: int = 52,
    ):
        self.n_estimators = n_estimators
        self.dist = dist
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.learning_rate = learning_rate
        self.random_state = random_state

    def inner_train(
        self, features: pd.DataFrame, target: pd.DataFrame, config: ModelConfig
    ) -> Estimator:
        """
        Uses a ngboost model to predict the energy rate values
        """
        ng = NGBRegressor(
            n_estimators=self.n_estimators,
            Dist=self.dist,
            verbose=self.verbose,
            verbose_eval=self.verbose_eval,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
        )
        X = features.values
        y: np.ndarray = target.values

        if y.shape[1] == 1:
            y = y.ravel()

        ng.fit(X, y)
        estimator = NGBoostEstimator(ng)
        return estimator
