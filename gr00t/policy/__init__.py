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

from .policy import BasePolicy, PolicyWrapper

# Lazy-import Gr00tPolicy: it pulls in `transformers` and the full backbone stack,
# which the lightweight inference client (PolicyClient over ZMQ) does not need.
# Clients running in a minimal venv (e.g., robomme_benchmark uv) can `from
# gr00t.policy.server_client import PolicyClient` without triggering heavy deps.
def __getattr__(name):
    if name == "Gr00tPolicy":
        from .gr00t_policy import Gr00tPolicy as _Gr00tPolicy
        return _Gr00tPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BasePolicy",
    "Gr00tPolicy",
    "PolicyWrapper",
]
