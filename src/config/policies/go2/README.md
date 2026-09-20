# Go2 locomotion policy

`policy.pt` is the TorchScript network that walks the Unitree Go2, and
`env.yaml` is the Isaac Lab environment config it was trained under. The
wrapper reads the physics rate, decimation, action scale, joint gains and effort
limits out of `env.yaml`, so the two files must always be replaced together.

## Why the wrapper ships its own policy

Isaac Sim 6.0 ships `Isaac/Samples/Policies/go2/physx_policy.pt`, and
`isaacsim.robot.policy.examples.robots.Go2FlatTerrainPolicy` loads it by
default. On Isaac Sim 6.0.1-rc.7 that policy stands but does not walk: any
non-zero `/cmd_vel` makes the robot splay its legs and drag. It was tested
against all three shipped Go2 USDs, on CPU and GPU physics, with both implicit
PhysX drives and explicit DCMotor torque, and at command speeds from 0.1 to
0.8 m/s. Nothing produced a gait.

Two other things were found along the way and are fixed in this package:

- **Wrong asset.** `go2.py` defaults to the Mujoco Menagerie conversion
  (`Isaac/Samples/Mujoco_Menagerie/unitree_go2/`). That asset is the Newton
  one — `spot.py` in the same extension picks between a Newton asset and a
  PhysX asset, and `go2.py` is missing that branch. The policy is trained
  against `Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd`, which is what
  `asset_paths.py` now resolves.
- **Wrong actuator.** `PolicyController` hands position targets to PhysX's
  implicit joint drives. Isaac Lab, where these policies are trained, uses an
  explicit DCMotor: it computes the torque in Python every physics step and
  applies it as an effort, with a velocity-dependent clamp that a plain
  symmetric effort limit does not reproduce. The Go2's `RobotSpec` sets
  `actuation="explicit"` so inference uses the same actuator as training.

## Regenerating

Requires Isaac Lab with `rsl_rl` (`$ISAACLAB_PATH`, `/workspace/isaaclab` in
the Docker image). Train:

```bash
cd "$ISAACLAB_PATH"
HEADLESS=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --headless --num_envs 4096 --max_iterations 1500
```

`HEADLESS=1` matters: the Docker image sets `HEADLESS=false` for Isaac Sim, and
Isaac Lab parses that variable as an integer and dies on it.

Then export into this directory:

```bash
python3 src/scripts/export_go2_policy.py
```

The exporter takes the newest checkpoint from the newest run, rebuilds the actor
MLP from the checkpoint's own tensor shapes, writes TorchScript, copies the
run's `params/env.yaml`, and refuses anything that is not a 48 → 12 flat-terrain
policy (the rough-terrain task's observation includes a 187-element height scan
that this wrapper has no sensor for).

## Iteration count matters more than reward suggests

Use the full 1500 iterations. A 300-iteration policy already reaches a mean
reward around 34 -- within 10% of the final 36 -- but walks in a deep crouch
(base height 0.14 m instead of 0.33 m) and tracks about a third of the commanded
velocity. Reward is a poor proxy for gait quality here, so judge a retrained
policy on base height and velocity tracking, not on the training curve.

Measured with the shipped 1500-iteration policy, commanding 0.5 m/s forward on
flat ground:

| Quantity | Value |
| --- | --- |
| Standing base height | 0.345 m |
| Walking base height | 0.326 m (peak-to-peak bob 0.008 m) |
| Steady-state body-frame `v_x` | 0.479 m/s against 0.50 commanded (96%) |

If a retrained policy misbehaves, the way to tell a policy problem from an
integration one is to run the same exported `policy.pt` inside Isaac Lab's own
`Isaac-Velocity-Flat-Unitree-Go2-Play-v0` environment and compare base height
and velocity. The two agreed to within a millimetre during bring-up, including
while the undertrained policy was crouching -- which is how the crouch was
identified as the policy's behaviour rather than a bug in this wrapper.
