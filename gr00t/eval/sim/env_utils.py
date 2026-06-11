# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gr00t.data.embodiment_tags import EmbodimentTag


# Mapping from gym-registered env_name prefix to EmbodimentTag.
# The prefix is the part before "/" in env_name (e.g. "libero_sim" from "libero_sim/task").
# Add new entries here when supporting a new benchmark.
ENV_PREFIX_TO_EMBODIMENT_TAG: dict[str, EmbodimentTag] = {
    # RoboCasa-Kitchen. Maps to NEW_EMBODIMENT; the modality config is registered at
    # runtime via the data config (single_panda_gripper).
    "robocasa_panda_omron": EmbodimentTag.NEW_EMBODIMENT,
}


def get_embodiment_tag_from_env_name(env_name: str) -> EmbodimentTag:
    """Get the EmbodimentTag for a gym-registered environment name.

    Looks up the env_name prefix (before "/") in ENV_PREFIX_TO_EMBODIMENT_TAG.
    Falls back to using the prefix directly as an EmbodimentTag value.
    """
    prefix = env_name.split("/")[0]
    if prefix in ENV_PREFIX_TO_EMBODIMENT_TAG:
        return ENV_PREFIX_TO_EMBODIMENT_TAG[prefix]
    return EmbodimentTag(prefix)
