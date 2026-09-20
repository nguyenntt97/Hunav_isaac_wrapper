#!/usr/bin/env python3
"""
export_go2_policy.py

Export an Isaac Lab / rsl_rl Go2 checkpoint to the TorchScript file this
package ships as config/policies/go2/policy.pt.

Why this exists: Isaac Sim 6.0's own Go2 policy
(Isaac/Samples/Policies/go2/physx_policy.pt) does not hold a usable gait, so the
wrapper carries its own. See config/policies/go2/README.md.

Runs under plain python3 -- it only needs torch, not Isaac Sim, because it reads
the checkpoint's tensors directly rather than reconstructing an rsl_rl runner.

Usage:
    python3 src/scripts/export_go2_policy.py [--run-dir DIR] [--out DIR]
"""

import argparse
import glob
import os
import shutil
import sys

import torch

DEFAULT_RUNS = "/workspace/isaaclab/logs/rsl_rl/unitree_go2_flat"
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "policies", "go2",
)

# The flat-terrain Go2 observation is 48 wide and the action is one target per
# leg joint. Asserting this catches an accidental export of the rough-terrain
# policy, whose observation carries a 187-element height scan the wrapper has no
# sensor for.
OBS_DIM = 48
ACTION_DIM = 12


def latest_run(runs_root):
    runs = sorted(glob.glob(os.path.join(runs_root, "*/")))
    if not runs:
        raise SystemExit(
            f"No training runs under {runs_root}. Train one first:\n"
            "  cd $ISAACLAB_PATH && HEADLESS=1 ./isaaclab.sh -p "
            "scripts/reinforcement_learning/rsl_rl/train.py "
            "--task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless "
            "--num_envs 4096 --max_iterations 1500"
        )
    return runs[-1]


def latest_checkpoint(run_dir):
    ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not ckpts:
        raise SystemExit(f"No model_*.pt in {run_dir}")
    return max(ckpts, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))


def build_actor(state_dict):
    """Rebuild the actor MLP from the checkpoint's own tensor shapes.

    Only the deterministic network is exported. ``distribution.std_param`` is
    the exploration noise PPO samples with during training; applying it at
    inference would make the robot twitch.
    """
    weight_keys = sorted(
        (k for k in state_dict if k.startswith("mlp.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    if not weight_keys:
        raise SystemExit(f"No mlp.*.weight tensors in checkpoint; got {list(state_dict)}")

    layers = []
    for index, weight_key in enumerate(weight_keys):
        weight = state_dict[weight_key]
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(state_dict[weight_key.replace(".weight", ".bias")])
        layers.append(linear)
        if index < len(weight_keys) - 1:
            # UnitreeGo2FlatPPORunnerCfg trains with ELU.
            layers.append(torch.nn.ELU())

    dims = [state_dict[weight_keys[0]].shape[1]] + [
        state_dict[k].shape[0] for k in weight_keys
    ]
    return torch.nn.Sequential(*layers).eval(), dims


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="Training run directory")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run(args.runs_root)
    checkpoint = latest_checkpoint(run_dir)
    print(f"run:        {run_dir}")
    print(f"checkpoint: {checkpoint}")

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor, dims = build_actor(blob["actor_state_dict"])
    print(f"layer dims: {dims}")
    if dims[0] != OBS_DIM or dims[-1] != ACTION_DIM:
        raise SystemExit(
            f"Expected a {OBS_DIM} -> {ACTION_DIM} flat-terrain policy, got "
            f"{dims[0]} -> {dims[-1]}. Is this the rough-terrain run?"
        )

    os.makedirs(args.out, exist_ok=True)
    policy_path = os.path.join(args.out, "policy.pt")
    torch.jit.script(actor).save(policy_path)

    # The env config travels with the policy: it is where the wrapper reads the
    # physics rate, decimation, action scale, joint gains and effort limits from.
    env_src = os.path.join(run_dir, "params", "env.yaml")
    shutil.copy(env_src, os.path.join(args.out, "env.yaml"))

    # Reload the way PolicyController does and confirm it is bit-identical.
    reloaded = torch.jit.load(policy_path)
    probe = torch.zeros(OBS_DIM)
    with torch.no_grad():
        drift = float((actor(probe) - reloaded(probe)).abs().max())
    if drift != 0.0:
        raise SystemExit(f"TorchScript round-trip changed the policy (max diff {drift})")

    print(f"wrote {policy_path} ({os.path.getsize(policy_path)} bytes)")
    print(f"wrote {os.path.join(args.out, 'env.yaml')}")
    print("round-trip verified identical")


if __name__ == "__main__":
    sys.exit(main())
