"""
ExpertPolicy: Uses the PPO-trained expert to control the vehicle.
Falls back to IDMPolicy when the expert fails (e.g., observation mismatch, toll gate).
"""
import numpy as np

from metadrive.examples import expert
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.engine.logger import get_logger

logger = get_logger()


class ExpertPolicy(BasePolicy):
    """
    Policy that uses the PPO expert (trained on standard roads).
    Falls back to IDM when expert fails (e.g., toll gate, unknown scenarios).
    """

    def __init__(self, control_object, random_seed=None, config=None):
        super(ExpertPolicy, self).__init__(control_object, random_seed, config)
        self._use_fallback = False
        self._fallback_policy = None

    def _get_fallback_policy(self):
        if self._fallback_policy is None:
            self._fallback_policy = IDMPolicy(
                self.control_object,
                random_seed=self.random_seed,
            )
            logger.warning("Expert failed — falling back to IDMPolicy for this episode.")
        return self._fallback_policy

    def act(self, agent_id=None):
        if self._use_fallback:
            return self._get_fallback_policy().act(agent_id)

        try:
            deterministic = self.config.get("deterministic", True)
            action = expert(self.control_object, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            self.action_info["action"] = action
            return action
        except (ValueError, AssertionError) as e:
            self._use_fallback = True
            logger.warning(
                "Expert policy failed (obs mismatch or unsupported scenario): %s. Using IDM fallback.",
                str(e)
            )
            return self._get_fallback_policy().act(agent_id)

    def reset(self):
        super(ExpertPolicy, self).reset()
        self._use_fallback = False
        if self._fallback_policy is not None:
            self._fallback_policy.reset()
