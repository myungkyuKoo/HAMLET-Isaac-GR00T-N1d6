# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modality config for RoboMME (Panda 7-DoF joint-angle action).
# Loaded via `--modality-config-path` and registered under `EmbodimentTag.NEW_EMBODIMENT`.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


robomme_panda_joint = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_view", "wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_position", "gripper_position"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
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


register_modality_config(robomme_panda_joint, EmbodimentTag.NEW_EMBODIMENT)
