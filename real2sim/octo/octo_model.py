from collections import defaultdict
import numpy as np
import os
import jax, cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from octo.model.octo_model import OctoModel

from sapien.core import Pose
from transforms3d.euler import euler2axangle
from collections import deque
from real2sim.utils.action.action_ensemble import ActionEnsembler
from transformers import AutoTokenizer

class OctoInference:
    def __init__(
        self,
        model_type="octo-base",
        dataset_id='bridge_dataset',
        policy_setup='widowx_bridge',
        image_size=256,
        action_scale=1.0,
        horizon=2,
        pred_action_horizon=4,
        exec_horizon=1,
        action_ensemble=True,
        action_ensemble_temp=0.0,
    ):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        if model_type in ['octo-base', 'octo-small']:
            self.model_type = f"hf://rail-berkeley/{model_type}"
            self.tokenizer, self.tokenizer_kwargs = None, None
            self.model = OctoModel.load_pretrained(self.model_type)
            self.action_mean = self.model.dataset_statistics[dataset_id]['action']['mean']
            self.action_std = self.model.dataset_statistics[dataset_id]['action']['std']
            self.automatic_task_creation = True
        else:
            self.model_type = model_type
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
            self.tokenizer_kwargs = {
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            }
            self.model = tf.saved_model.load(self.model_type)
            assert dataset_id in ['bridge_dataset']
            self.action_mean = np.array([
                0.00021161, 0.00012614, -0.00017022, -0.00015062, -0.00023831, 0.00025646, 0.0
            ])
            self.action_std = np.array([
                0.00963721, 0.0135066, 0.01251861, 0.02806791, 0.03016905, 0.07632624, 1.0
            ])
            self.automatic_task_creation = False
        
        self.policy_setup = policy_setup
        assert self.policy_setup in ['widowx_bridge']
        
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp
        
        self.goal_gripper_pose_at_robot_base = None
        self.goal_gripper_closedness = np.array([0.0])
        self.task = None
        self.image_history = deque(maxlen=self.horizon)
        self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        self.num_image_history = 0
        self.time_step = 0

    def _resize_image(self, image):
        # image = cv2.resize(image, (self.image_size, self.image_size))
        image = tf.image.resize(
            image, size=(self.image_size, self.image_size), method="lanczos3", antialias=True
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image
        
    def _add_image_to_history(self, image):
        if self.num_image_history == 0:
            self.image_history.extend([image] * self.horizon)
        else:
            self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)
        
    def _obtain_image_history_and_mask(self):
        images = np.stack(self.image_history, axis=0)
        pad_mask = np.ones(self.horizon, dtype=bool)
        pad_mask[:self.horizon - self.num_image_history] = 0
        return images, pad_mask
        
    def reset(self, task_description):
        if self.automatic_task_creation:
            self.task = self.model.create_tasks(texts=[task_description])
        else:
            self.task = self.tokenizer(task_description, **self.tokenizer_kwargs)
        self.image_history.clear()
        self.action_ensembler.reset()
        self.num_image_history = 0
        self.time_step = 0

    def step(self, image, *args, **kwargs):
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]
        
        if self.automatic_task_creation:
            input_observation = {
                'image_primary': images,
                'pad_mask': pad_mask
            }
            norm_raw_actions = self.model.sample_actions(input_observation, self.task, rng=jax.random.PRNGKey(0))
        else:
            input_observation = {
                'image_primary': images,
                'timestep_pad_mask': pad_mask
            }
            input_observation = {
                'observations': input_observation,
                'tasks': {
                    'language_instruction': self.task
                },
                'rng': np.zeros((4,), dtype=np.uint32),
            }
            norm_raw_actions = self.model.lc_ws2(input_observation)[:, :, :7]
        norm_raw_actions = norm_raw_actions[0]   # remove batch, becoming (action_pred_horizon, action_dim)
        
        if self.action_ensemble:
            norm_raw_actions = self.action_ensembler.ensemble_action(norm_raw_actions)
            norm_raw_actions = norm_raw_actions[None] # [1, 7]
            
        raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]), # range [0, 1]; 1 = open; 0 = close
        }
        
        # process raw_action to obtain the action to be sent to the environment
        action = {}
        action['world_vector'] = raw_action['world_vector'] * self.action_scale
        action_rotation_delta = np.asarray(raw_action['rotation_delta'], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action['rot_axangle'] = action_rotation_axangle * self.action_scale
        
        action['gripper'] = 2.0 * (raw_action['open_gripper'] > 0.5) - 1.0 # binarize gripper action to be -1 or 1
        action['terminate_episode'] = np.array([0.0])
        
        self.time_step += 1
        
        return raw_action, action
    
    def visualize_epoch(self, predicted_raw_actions, images, save_path):
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [
            ['image'] * len(ACTION_DIM_LABELS),
            ACTION_DIM_LABELS
        ]
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array([np.concatenate([a['world_vector'], a['rotation_delta'], a['open_gripper']], axis=-1) for a in predicted_raw_actions])
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label='predicted action')
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel('Time in one episode')

        axs['image'].imshow(img_strip)
        axs['image'].set_xlabel('Time in one episode (subsampled)')
        plt.legend()
        plt.savefig(save_path)
        



"""
Bridge:
self.action_mean = np.array([
    0.00021161, 0.00012614, -0.00017022, -0.00015062, -0.00023831, 0.00025646, 0.0
])
self.action_std = np.array([
    0.00963721, 0.0135066, 0.01251861, 0.02806791, 0.03016905, 0.07632624, 1.0
])
"""