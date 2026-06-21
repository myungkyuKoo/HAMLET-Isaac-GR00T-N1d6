# RMBench policy plugin for the GR00T N1.6 (+ HAMLET) ZMQ policy server.
# RMBench's script/eval_policy.py loads this via `sys.path.append("./policy")` + import_module(policy_name),
# so this whole `policy/` dir is symlinked into <RMBENCH_ROOT>/policy/<name>/ by run_scripts/rmbench/eval_n1d6_rmbench.sh.
import os, sys
sys.path.append(os.path.dirname(__file__))
from rmbench_bridge import ServerBridgeModel, run_eval

def get_model(usr_args):
    host = str(usr_args.get("server_host", os.environ.get("RM_SERVER_HOST", "127.0.0.1")))
    port = int(usr_args.get("server_port", os.environ.get("RM_SERVER_PORT", 5555)))
    exec_chunk = int(usr_args.get("exec_chunk", 16))
    # Single current frame per view; HAMLET memory is server-side (session_ids/reset_memory) and
    # accumulates over the episode — no demo-frame priming (RMBench has no is_demo/observe phase).
    return ServerBridgeModel(host=host, port=port, video_deltas=(0,), exec_chunk=exec_chunk)

def eval(TASK_ENV, model, observation):
    run_eval(TASK_ENV, model, observation)

def reset_model(model):
    model.reset()
