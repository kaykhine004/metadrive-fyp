import os
import numpy as np
from metadrive.policy.base_policy import BasePolicy

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "test", "Final analysis", "Intersection analysis", "optimized_policy_model.zip"
)


class OptimizedPolicy(BasePolicy):
    """
    PPO-trained policy optimized for the Roundabout + Intersection map.
    Trained to minimize crashes while reaching the destination quickly.
    """
    _model = None

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            from stable_baselines3 import PPO
            resolved = os.path.normpath(MODEL_PATH)
            assert os.path.exists(resolved), f"Trained model not found at {resolved}"
            cls._model = PPO.load(resolved, device="cpu")
        return cls._model

    def __init__(self, control_object, random_seed):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self.model = self._load_model()

    def act(self, agent_id=None):
        obs = self.control_object.get_state()
        obs_array = np.array(list(obs.values()), dtype=np.float32) if isinstance(obs, dict) else obs

        from metadrive.obs.state_obs import LidarStateObservation
        try:
            obs_from_env = self.engine.observations[self.control_object.name].observe(self.control_object)
        except Exception:
            obs_from_env = obs_array

        action, _ = self.model.predict(obs_from_env, deterministic=True)
        action = [float(action[0]), float(action[1])]
        self.action_info["action"] = action
        return action
