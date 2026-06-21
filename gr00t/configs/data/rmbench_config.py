# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modality config for RMBench (RoboTwin-2.0 dual-arm Aloha-AgileX; 14-D absolute joint action).
# Loaded via `--modality-config-path` and registered under `EmbodimentTag.NEW_EMBODIMENT`.
#
# RMBench is a dual-arm benchmark: each arm is 6-DoF + 1 gripper. State/action are 14-D **absolute
# joint** targets laid out contiguously as [L_arm(6), R_arm(6), L_grip, R_grip] -> joint_position 0:12,
# gripper 12:14 (see meta/modality.json). Three camera views: front (overhead), left wrist, right wrist.
# Action horizon is 50 (GR00T-N1.6 native); the eval executes the first `n_action_steps` (=16) per call.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


rmbench_aloha_joint = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_view", "left_wrist_view", "right_wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_position", "gripper_position"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(50)),
        modality_keys=["joint_position", "gripper_close"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


register_modality_config(rmbench_aloha_joint, EmbodimentTag.NEW_EMBODIMENT)
