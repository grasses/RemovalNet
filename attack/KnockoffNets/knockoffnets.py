# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Knockoff Nets attack `KnockoffNets`.

| Paper link: https://arxiv.org/abs/1812.02766
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import torch
import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import ExtractionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import to_categorical
import torch.nn.functional as F

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class KnockoffNets(ExtractionAttack):
    """
    Implementation of the Knockoff Nets attack from Orekondy et al. (2018).

    | Paper link: https://arxiv.org/abs/1812.02766
    """

    attack_params = ExtractionAttack.attack_params + [
        "batch_size_fit",
        "batch_size_query",
        "nb_epochs",
        "nb_stolen",
        "sampling_strategy",
        "reward",
        "verbose",
        "use_probability",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        batch_size_fit: int = 1,
        batch_size_query: int = 1,
        nb_epochs: int = 10,
        nb_stolen: int = 1,
        nb_classes: int = 1000,
        sampling_strategy: str = "random",
        reward: str = "all",
        verbose: bool = True,
        use_probability: bool = False,
    ) -> None:
        """
        Create a KnockoffNets attack instance. Note, it is assumed that both the victim classifier and the thieved
        classifier produce logit outputs.

        :param classifier: A victim classifier.
        :param batch_size_fit: Size of batches for fitting the thieved classifier.
        :param batch_size_query: Size of batches for querying the victim classifier.
        :param nb_epochs: Number of epochs to use for training.
        :param nb_stolen: Number of queries submitted to the victim classifier to steal it.
        :param sampling_strategy: Sampling strategy, either `random` or `adaptive`.
        :param reward: Reward type, in ['cert', 'div', 'loss', 'all'].
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)

        self.batch_size_fit = batch_size_fit
        self.batch_size_query = batch_size_query
        self.nb_epochs = nb_epochs
        self.nb_stolen = nb_stolen
        self.nb_classes = nb_classes
        self.sampling_strategy = sampling_strategy
        self.reward = reward
        self.verbose = verbose
        self.use_probability = use_probability
        self._check_params()
        self.step = 1
        self.print_step = int(nb_stolen / 2)

        self.h_func = np.zeros(nb_classes)
        self.learning_rate = np.zeros(nb_classes)
        self.probs = np.ones(nb_classes) / nb_classes
        self.reward_avg = np.zeros(3)
        self.reward_var = np.zeros(3)

    def extract(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> "CLASSIFIER_TYPE":
        """
        Extract a thieved classifier.

        :param x: An array with the source input to the victim classifier.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  `(nb_samples,)`.
        :param thieved_classifier: A thieved classifier to be stolen.
        :return: The stolen classifier.
        """
        # Check prerequisite for random strategy
        if self.sampling_strategy == "random" and y is not None:
            logger.warning("This attack with random sampling strategy does not use the provided label y.")

        # Check prerequisite for adaptive strategy
        if self.sampling_strategy == "adaptive" and y is None:
            raise ValueError("This attack with adaptive sampling strategy needs label y.")

        # Check the size of the source input vs nb_stolen
        if x.shape[0] < self.nb_stolen:
            logger.warning(
                f"The size of the source input {x.shape[0]} < {self.nb_stolen} is smaller than the expected number of queries submitted "
                "to the victim classifier."
            )

        # Check if there is a thieved classifier provided for training
        thieved_classifier = kwargs.get("thieved_classifier")
        if thieved_classifier is None or not isinstance(thieved_classifier, ClassifierMixin):
            raise ValueError("A thieved classifier is needed.")

        # Implement model extractions
        if self.sampling_strategy == "random":
            thieved_classifier = self._random_extraction(x, thieved_classifier)  # type: ignore
        else:
            thieved_classifier = self._adaptive_extraction(x, y, thieved_classifier)  # type: ignore
        self.step += 1
        return thieved_classifier

    def _random_extraction(self, x: np.ndarray, thieved_classifier: "CLASSIFIER_TYPE") -> "CLASSIFIER_TYPE":
        """
        Extract with the random sampling strategy.

        :param x: An array with the source input to the victim classifier.
        :param thieved_classifier: A thieved classifier to be stolen.
        :return: The stolen classifier.
        """
        # Select data to attack
        selected_x = self._select_data(x)

        # Query the victim classifier
        fake_labels = self._query_label(selected_x)

        # Train the thieved classifier
        thieved_classifier.fit(
            x=selected_x,
            y=fake_labels,
            batch_size=self.batch_size_fit,
            nb_epochs=self.nb_epochs,
            verbose=0,
        )

        return thieved_classifier

    def _select_data(self, x: np.ndarray) -> np.ndarray:
        """
        Select data to attack.

        :param x: An array with the source input to the victim classifier.
        :return: An array with the selected input to the victim classifier.
        """
        nb_stolen = np.minimum(self.nb_stolen, x.shape[0])
        rnd_index = np.random.choice(x.shape[0], nb_stolen, replace=False)

        return x[rnd_index].astype(ART_NUMPY_DTYPE)

    def _query_label(self, x: np.ndarray) -> np.ndarray:
        """
        Query the victim classifier.

        :param x: An array with the source input to the victim classifier.
        :return: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        """
        labels = self.estimator.predict(x=x, batch_size=self.batch_size_query)
        if not self.use_probability:
            labels = np.argmax(labels, axis=1)
            labels = to_categorical(labels=labels, nb_classes=self.estimator.nb_classes)
        return labels

    def _adaptive_extraction(
        self, x: np.ndarray, y: np.ndarray, thieved_classifier: "CLASSIFIER_TYPE",
            use_global: bool=False
    ) -> "CLASSIFIER_TYPE":
        """
        Extract with the adaptive sampling strategy.

        :param x: An array with the source input to the victim classifier.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param thieved_classifier: A thieved classifier to be stolen.
        :return: The stolen classifier.
        """
        # Compute number of actions
        '''
        if len(y.shape) == 2:
            nb_actions = len(np.unique(np.argmax(y, axis=1)))
        elif len(y.shape) == 1:
            nb_actions = len(np.unique(y))
        else:
            raise ValueError("Target values `y` has a wrong shape.")
        '''

        nb_actions = self.nb_classes
        # We need to keep an average version of the victim output
        if self.reward == "div" or self.reward == "all":
            self.y_avg = np.zeros(self.estimator.nb_classes)

        # We need to keep an average and variance version of rewards
        if self.reward == "all" and not use_global:
            self.reward_avg = np.zeros(3)
            self.reward_var = np.zeros(3)

        # Implement the bandit gradients algorithm
        if not use_global:
            self.h_func = np.zeros(nb_actions)
            self.learning_rate = np.zeros(nb_actions)
            self.probs = np.ones(nb_actions) / nb_actions

        selected_x = []
        selected_y = []
        selected_idx = []
        queried_labels = []

        avg_reward = 0.0
        for iteration in trange(1, self.nb_stolen + 1, desc="Knock-off nets", disable=not self.verbose):
            # Sample an action, nb_actions=nb_classes
            try:
                action = np.random.choice(np.arange(0, nb_actions), p=self.probs)
                # Sample data to attack
                idx, sampled_x = self._sample_data(x, y, action, choice=True)
                if sampled_x is None:
                    continue
                sampled_x = np.expand_dims(sampled_x, axis=0)
                selected_x.append(sampled_x)
                selected_y.append(action)
                selected_idx.append(idx)
            except Exception as e:
                print(f"-> sample stop after {iteration} iteration!!\n")
                break

            # Query the victim classifier
            y_logit = self.estimator.predict(x=sampled_x, batch_size=self.batch_size_query)
            y_output = F.softmax(torch.from_numpy(y_logit), dim=1).cpu().numpy()
            fake_label = np.argmax(y_output, axis=1)
            fake_label = to_categorical(labels=fake_label, nb_classes=self.estimator.nb_classes)
            queried_labels.append(fake_label[0])

            # Train the thieved classifier
            thieved_classifier.fit(
                x=sampled_x,
                y=fake_label,
                batch_size=self.batch_size_fit,
                nb_epochs=1,
                verbose=0,
            )

            # Test new labels
            y_hat = thieved_classifier.predict(x=sampled_x, batch_size=self.batch_size_query)
            y_hat = F.softmax(torch.from_numpy(y_hat), dim=1).cpu().numpy()

            # Compute rewards
            reward = self._reward(y_output, y_hat, iteration)
            avg_reward = avg_reward + (1.0 / iteration) * (reward - avg_reward)

            # Update learning rate
            self.learning_rate[action] += 1

            # Update H function
            for i_action in range(nb_actions):
                if i_action != action:
                    self.h_func[i_action] = (
                        self.h_func[i_action] - 1.0 / self.learning_rate[action] * (reward - avg_reward) * self.probs[i_action]
                    )
                else:
                    self.h_func[i_action] = self.h_func[i_action] + 1.0 / self.learning_rate[action] * (reward - avg_reward) * (
                        1 - self.probs[i_action]
                    )
            # Update probs
            aux_exp = np.exp(self.h_func)
            self.probs = aux_exp / (np.sum(aux_exp) + 1e-8)

        if self.step % 10 == 0:
            print(f"-> {np.sort(self.probs)[-20:]}\n-> prob ∈ [{min(self.probs)} ~ {max(self.probs)}] size:{len(selected_x)}\n")

        # Train the thieved classifier the final time
        thieved_classifier.fit(
            x=np.concatenate(selected_x, axis=0),
            y=np.array(queried_labels),
            batch_size=self.batch_size_fit,
            nb_epochs=self.nb_epochs,
        )
        return (selected_idx, selected_x, selected_y), self.probs, thieved_classifier

    @staticmethod
    def _sample_data(x: np.ndarray, y: np.ndarray, action: int, choice: bool=False) -> np.ndarray:
        """
        Sample data with a specific action.

        :param x: An array with the source input to the victim classifier.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param action: The action index returned from the action sampling.
        :return: An array with one input to the victim classifier.
        """
        if len(y.shape) == 2:
            y_index = np.argmax(y, axis=1)
        else:
            y_index = y

        bool_idx = y_index == action
        x_index = x[bool_idx]

        #print(action, y_index, bool_idx)

        if len(x_index) <= 0:
            if choice:
                return 0, None
            else:
                return None

        rnd_idx = np.random.choice(len(x_index))
        if choice:
            idx = np.argsort(-1 * bool_idx)[rnd_idx]
            return idx, x_index[rnd_idx]
        else:
            return x_index[rnd_idx]

    def _reward(self, y_output: np.ndarray, y_hat: np.ndarray, n: int) -> float:
        """
        Compute reward value.

        :param y_output: Output of the victim classifier.
        :param y_hat: Output of the thieved classifier.
        :param n: Current iteration.
        :return: Reward value.
        """
        if self.reward == "cert":
            return self._reward_cert(y_output)
        if self.reward == "div":
            return self._reward_div(y_output, n)
        if self.reward == "loss":
            return self._reward_loss(y_output, y_hat)
        return self._reward_all(y_output, y_hat, n)

    @staticmethod
    def _reward_cert(y_output: np.ndarray) -> float:
        """
        Compute `cert` reward value.

        :param y_output: Output of the victim classifier.
        :return: Reward value.
        """
        largests = np.partition(y_output.flatten(), -2)[-2:]
        reward = largests[1] - largests[0]

        return reward

    def _reward_div(self, y_output: np.ndarray, n: int) -> float:
        """
        Compute `div` reward value.

        :param y_output: Output of the victim classifier.
        :param n: Current iteration.
        :return: Reward value.
        """
        # First update y_avg
        self.y_avg = self.y_avg + (1.0 / n) * (y_output[0] - self.y_avg)

        # Then compute reward
        reward = 0
        for k in range(self.estimator.nb_classes):
            reward += np.maximum(0, y_output[0][k] - self.y_avg[k])

        return reward

    def _reward_loss(self, y_output: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Compute `loss` reward value.

        :param y_output: Output of the victim classifier.
        :param y_hat: Output of the thieved classifier.
        :return: Reward value.
        """
        # Compute victim probs
        aux_exp = np.exp(y_output[0])
        probs_output = aux_exp / np.sum(aux_exp)

        # Compute thieved probs
        aux_exp = np.exp(y_hat[0])
        probs_hat = aux_exp / np.sum(aux_exp)

        # Compute reward
        reward = 0
        for k in range(self.estimator.nb_classes):
            reward += -probs_output[k] * np.log(probs_hat[k])

        return reward

    def _reward_all(self, y_output: np.ndarray, y_hat: np.ndarray, n: int) -> np.ndarray:
        """
        Compute `all` reward value.

        :param y_output: Output of the victim classifier.
        :param y_hat: Output of the thieved classifier.
        :param n: Current iteration.
        :return: Reward value.
        """
        reward_cert = self._reward_cert(y_output)
        reward_div = self._reward_div(y_output, n)
        reward_loss = self._reward_loss(y_output, y_hat)
        reward = [reward_cert, reward_div, reward_loss]
        self.reward_avg = self.reward_avg + (1.0 / n) * (reward - self.reward_avg)
        self.reward_var = self.reward_var + (1.0 / n) * ((reward - self.reward_avg) ** 2 - self.reward_var)

        # Normalize rewards
        if n > 1:
            reward = (reward - self.reward_avg) / np.sqrt(self.reward_var)
        else:
            reward = [max(min(r, 1), 0) for r in reward]

        return np.mean(reward)

    def _check_params(self) -> None:
        if not isinstance(self.batch_size_fit, (int, np.int)) or self.batch_size_fit <= 0:
            raise ValueError("The size of batches for fitting the thieved classifier must be a positive integer.")

        if not isinstance(self.batch_size_query, (int, np.int)) or self.batch_size_query <= 0:
            raise ValueError("The size of batches for querying the victim classifier must be a positive integer.")

        if not isinstance(self.nb_epochs, (int, np.int)) or self.nb_epochs <= 0:
            raise ValueError("The number of epochs must be a positive integer.")

        if not isinstance(self.nb_stolen, (int, np.int)) or self.nb_stolen <= 0:
            raise ValueError("The number of queries submitted to the victim classifier must be a positive integer.")

        if self.sampling_strategy not in ["random", "adaptive"]:
            raise ValueError("Sampling strategy must be either `random` or `adaptive`.")

        if self.reward not in ["cert", "div", "loss", "all"]:
            raise ValueError("Reward type must be in ['cert', 'div', 'loss', 'all'].")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
        if not isinstance(self.use_probability, bool):
            raise ValueError("The argument `use_probability` has to be of type bool.")