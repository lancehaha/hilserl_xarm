# HIL-SERL on a UFACTORY xArm7 with a SpaceMouse

Train an xArm7 on a real-world task with LeRobot's [HIL-SERL](https://hil-serl.github.io/) implementation:

1. record demonstrations with a 3Dconnexion SpaceMouse;
2. train a reward classifier;
3. train a policy online with SAC, taking over with the SpaceMouse whenever it needs correcting;
4. run the learned policy on its own.

Read [`docs/source/hilserl.mdx`](../../docs/source/hilserl.mdx) for the general method. This guide covers what is specific to the xArm setup.

## Training experiment plan

The first goal is to establish that the standard xArm HIL-SERL workflow learns the task. Keep the
SAC learner and the existing sparse success reward fixed while comparing replay and data-collection
choices. These ideas are inspired by [RAGEN's study of informative and diverse rollouts](https://arxiv.org/abs/2504.20073),
but are adaptations for a physical robot, not a reproduction of RAGEN's PPO/GRPO method.

1. **Define reproducible starts.** Mark a small, safe set of object starting poses within the
   camera view and robot workspace. Record the start-pose ID, policy version, trial outcome,
   intervention count, and episode length for each trial. Hold a few poses aside for evaluation.
2. **Collect varied, fresh trials.** Rotate among the training poses rather than resetting to one
   location. After policy updates, collect new trials with the current policy so replay continues
   to include its present mistakes. Keep demonstrations and older trials available for coverage.
3. **Use both replay routes.** Reserve separate shares of each online batch for (a) weighted
   individual actions from successful attempts, including gripper command events, and (b) a sweep
   through the transitions of one complete successful attempt. Select complete attempts with a
   higher weight for the shorter half, while retaining longer successes. Start a new sweep at its
   configured batch interval once the previous sweep has finished. Keep a uniform share for
   failures and recovery states. Both routes still train SAC with one-step transition targets;
   shorter trajectories receive sampling weight, not extra reward.
4. **Try uncertain-start weighting only when data supports it.** For a start pose with several
   trials, prioritize it modestly when outcomes are mixed (some successes and some failures).
   Do not infer uncertainty from one attempt, and do not discard always-failing poses; they can
   contain useful human recovery data. Record start-pose IDs in replay before implementing this
   sampler, then compare it against the existing episode-aware mixer.
5. **Evaluate with a frozen policy.** Run the same held-out start poses without intervention for
   each training variant and report success rate as the primary metric. Also report trial count,
   intervention count during training, and episode length. Use the same task and evaluation poses
   for every variant so any difference can be attributed to the training change.

This plan does not add intermediate grasp rewards or whole-trajectory policy optimization. Add
those only as separate experiments if the basic setup works and there is a reliable grasp signal
or evidence that one-step SAC fails to propagate the final success reward.

## How it works

The setup follows [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl) (`franka_env`, `RelativeFrame`, the `usb_pickup_insertion` / `ram_insertion` configs) as closely as the xArm allows:

| hil-serl (Franka)                                                           | Here (xArm7)                                                                                                     |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 7-D action: xyz, rotation vector, gripper, scaled by `ACTION_SCALE`         | Same: `end_effector_step_sizes` (m) and `end_effector_rot_step` (rad), discrete gripper                          |
| `RelativeFrame`: actions in the end-effector frame                          | `action_frame: "ee"`; SpaceMouse interventions are converted to that frame before they are stored                |
| State `tcp_pose` (relative to reset, Euler), `tcp_vel`, `tcp_force`, `tcp_torque`, `gripper_pose` | Same 19-D state (`relative_ee_frame`), force / torque from the UFACTORY Six-axis F/T sensor |
| Cartesian impedance controller (2000 N/m), reference clipping              | Controller F/T admittance mode (2000 N/m, 20 Nm/rad, the SDK maximum) + `reference_clip`                         |
| Stiff `PRECISION_PARAM` while resetting, compliant `COMPLIANCE_PARAM` after | Admittance off while homing; the F/T sensor is zeroed at the start pose, then admittance is switched on          |
| `ABS_POSE_LIMIT_LOW/HIGH` safety box, `RANDOM_XY_RANGE`, `RANDOM_RZ_RANGE`  | `end_effector_bounds` (xyz + roll / pitch / yaw, roll by magnitude), `random_xy_range_m`, `random_rz_range_rad` |

HIL-SERL learns **normalized end-effector deltas**: `[dx, dy, dz, drx, dry, drz] ∈ [-1, 1]` in the end-effector frame, plus a discrete gripper command `{0: close, 1: stay, 2: open}`. Each of these uses that same action format:

- the policy;
- the SpaceMouse;
- the recorded demos;
- the replay buffers.

The robot never sees a delta. On every step the action pipeline in `lerobot.rl.gym_manipulator` converts it into an absolute target:

```
SpaceMouse (base frame) ── TeleopDeltaToEEFrameStep ──┐ (while intervening)
policy ───────────────────────────────────────────────┤
   │  [dx, dy, dz, drx, dry, drz, gripper], EE frame   ← stored in the replay buffer / dataset, learned by SAC
   ▼
CartesianDeltaToTargetStep
   │  delta rotated into the base frame
   │  position = measured + delta × end_effector_step_sizes
   │  rotation = Rotation.from_rotvec(delta_rot × end_effector_rot_step) * measured
   │  clipped to end_effector_bounds, then to reference_clip around the measured pose
   ▼
XArm.send_action({"ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz", "gripper.pos"})  ← absolute target, base frame
   ▼
xArm controller (online Cartesian trajectory planning, mode 7, with F/T admittance) does the inverse kinematics
```

`ee.rx / ry / rz` are the xArm roll / pitch / yaw in radians. The gripper is binary: "stay" keeps the last open/close command, so a grasp keeps squeezing.

The policy state (`observation.state`, 19 values) is built by `EERelativeObservationStep` from `XArm.get_observation`: TCP pose relative to the pose after the reset (xyz, then roll / pitch / yaw), TCP twist in the end-effector frame (finite difference), external force and torque in the base frame, gripper opening. The reward classifier only sees the camera images.

Code involved:

| What                                           | Where                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------- |
| xArm robot (`--robot.type=xarm`)               | `src/lerobot/robots/xarm/xarm.py`, `config_xarm.py`                             |
| Delta → Cartesian target, EE frame, relative state | `src/lerobot/robots/xarm/robot_cartesian_processor.py`                      |
| HIL-SERL env, recording                        | `src/lerobot/rl/gym_manipulator.py` (generalized to robots without a motor bus) |
| SpaceMouse teleop (`--teleop.type=spacemouse`) | `src/lerobot/teleoperators/space_mouse/`                                        |
| Teleop + bounds tool, policy evaluation        | `examples/xarm_hilserl/teleoperate.py`, `eval_policy.py`                        |
| F/T sensor and admittance checks               | `examples/xarm_hilserl/probe_admittance.py`                                     |
| Configs                                        | `examples/xarm_hilserl/configs/`                                                |

## Hardware and I/O sources

| Device                                                               | Interface                         | Library           | Used for                                                                      |
| -------------------------------------------------------------------- | --------------------------------- | ----------------- | ----------------------------------------------------------------------------- |
| xArm7 + control box                                                  | Ethernet, `192.168.1.235`         | `xarm-python-sdk` | Arm motion, TCP pose                                                          |
| UFACTORY Six-axis F/T sensor (between flange and gripper)            | Tool bus, through the xArm SDK    | `xarm-python-sdk` | Force / torque state, admittance (compliance) control                         |
| xArm Gripper                                                         | Tool Modbus, through the xArm SDK | `xarm-python-sdk` | Open / close, opening read-back                                               |
| 3Dconnexion SpaceMouse Compact                                       | USB HID (`/dev/hidraw*`)          | `pyspacemouse`    | Arm xyz + rotation + gripper buttons                                          |
| Intel RealSense D435i, on the wrist (serial `134322070906`)          | USB 3                             | `pyrealsense2`    | `wrist` image: close view of the gripper and object (RGB, depth not used)     |
| Intel RealSense D405, fixed beside the table (serial `261122273645`) | USB 3                             | `pyrealsense2`    | `side` image: the whole scene, so the object is visible wherever it is placed |
| Keyboard                                                             | X11 session                       | `pynput`          | Intervention, success / failure, re-record                                    |
| GPU                                                                  | CUDA                              | PyTorch           | Learner (SAC updates) and actor (policy inference)                            |

> [!WARNING]
> `pynput` captures the keyboard **globally**: the keys below trigger events whichever window has focus. Don't type in other windows while a session is running.

## Install and one-time setup

1. **Python 3.12 environment.** `pyrealsense2` only ships Linux wheels for Python 3.12 / 3.13.

   ```bash
   uv sync --locked --extra xarm_hilserl --extra training --extra test
   ```

   The `xarm_hilserl` extra installs:
   - `lerobot[xarm]` (`xarm-python-sdk`);
   - `lerobot[spacemouse]`;
   - `lerobot[intelrealsense]`;
   - `lerobot[hilserl]`;
   - `lerobot[viz]`.

   `xarm-python-sdk` is not part of the base `lerobot` install.

2. **System libraries and device permissions.**
   - SpaceMouse: `pyspacemouse` loads the system `hidapi` library, which `uv` does not install. Without it, `teleop.connect()` fails with `HID API is probably not installed`:
     ```bash
     sudo apt install libhidapi-hidraw0
     ```
   - RealSense: install Intel's [udev rules](https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules).
   - SpaceMouse: if `pyspacemouse` cannot open the device, allow access to 3Dconnexion HID devices:
     ```bash
     echo 'KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"' | sudo tee /etc/udev/rules.d/99-spacemouse.rules
     sudo udevadm control --reload-rules && sudo udevadm trigger
     ```
3. **Cameras.** The configs select each camera by serial number. Both must be on USB 3 ports and cables (USB 2 drops frames).
   - `rs-enumerate-devices -s` lists the connected RealSense cameras and their serials.
   - `realsense-viewer` shows them live, which is useful for positioning and for trying `exposure` / `gain`.
   - `uv run lerobot-find-cameras realsense` saves a few frames per camera to `outputs/captured_images/`, exactly as lerobot sees them.
   - A camera can only be opened by one program at a time. Close the viewer before running any lerobot command.
4. **UFACTORY Studio.** These are controller settings, persistent and enforced by the controller even if Python crashes:
   - **TCP offset and payload** of the xArm Gripper, so the reported pose is the fingertip and collision detection is accurate.
   - **Safety boundary**: slightly larger than the soft box (`end_effector_bounds`) you'll find in step 1.
   - **Collision sensitivity** and **self-collision detection**.
5. **F/T sensor.** Run the sensor **load identification** in UFACTORY Studio (Settings → F/T sensor) with the gripper mounted. The arm moves through several poses on its own, so clear the workspace. Identification resets the TCP payload and collision sensitivity: set both again afterwards. Then check the sensor with the arm stiff:

   ```bash
   uv run python examples/xarm_hilserl/probe_admittance.py status   # sensor error 0, compensated force ≈ 0 N at rest
   uv run python examples/xarm_hilserl/probe_admittance.py frame    # push the gripper: +X → +Fx; up → -Fz (tool frame)
   uv run python examples/xarm_hilserl/probe_admittance.py track    # admittance on: a ~20 N push moves the arm ~1 cm, it comes back
   ```

   `ft_ext_force` is in the tool frame (with the gripper pointing down, pushing up reads -Fz). `robot.ft_sensor_frame: "tool"` uses it as is, as hil-serl does with the Franka `K_F_ext_hat_K`; `"base"` rotates it into the base frame. Datasets recorded before this change have base-frame force, convert them with `convert_ft_frame.py`. The sensor is zeroed after every reset, so a contact or cable pull at reset time shifts the zero for the whole episode.

## Controls reference

**SpaceMouse** (every process):

| Input            | Action                                                                                                                                        |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Puck translation | Move the TCP in x / y / z of the robot base frame (+X out of the front of the base, +Y to its left, +Z up). Remap with `teleop.axis_mapping`. |
| Puck tilt / twist | Rotate the TCP about the base x / y / z axes (`teleop.use_rotation`). The rotation follows `teleop.axis_mapping`: tilting / twisting the puck about the axis you push along rotates the TCP about the robot axis that push moves along. Override with `teleop.rotation_mapping`; small tilts below `teleop.rotation_deadzone` are ignored. |
| Left button      | Close the gripper (stays closed)                                                                                                              |
| Right button     | Open the gripper (stays open)                                                                                                                 |

**Keyboard**, per process:

| Process                                   | Space                                                      | Enter                                                 | Backspace            | `r`               | Ctrl+C                           |
| ----------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------- | -------------------- | ----------------- | -------------------------------- |
| `teleoperate.py`                          | —                                                          | Clear a fault and move home (when asked)              | —                    | —                 | Quit and print the config values |
| Dry run (`env.json`)                      | Latch intervention on / off                                | End episode: success                                  | End episode: failure | End episode: redo | Quit                             |
| Demo recording (`env.json --mode=record`) | Latch intervention on / off                                | End + save: success                                   | End + save: failure  | Discard + redo    | Quit                             |
| Classifier recording (see step 5)         | Latch intervention on / off                                | Press at success: every later frame labeled success   | End + save           | Discard + redo    | Quit                             |
| Online training (actor)                   | Latch intervention on / off                                | End episode: success                                  | End episode: failure | End episode       | Quit actor / learner             |
| Evaluation (`eval_policy.py`)             | Latch a safety takeover on / off                           | End episode: success                                  | End episode: failure | End episode       | Quit, print results              |

**Intervention**, as in rail-berkeley/hil-serl: you intervene on every step where you move the puck (beyond `teleop.deadzone`) or hold a gripper button; on the other steps the policy acts (or, without a policy, the robot holds still). Only the steps with input are labeled as interventions, so letting go of the SpaceMouse hands control back right away and pauses are not recorded as expert "hold still" actions. Press **Space** to latch the intervention on, e.g. to make the arm hold still against the policy; it switches back **off** at the end of every episode. Set `teleop.intervene_on_input=false` for the toggle-only behavior.

## Step by step

All commands run from the repository root, through `uv run` so they use the project environment. There are three configs in `examples/xarm_hilserl/configs/`:

| Config                         | Used by                                                                                               |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `env.json`                     | `gym_manipulator`: dry run, demo recording and classifier recording, switched with command-line flags |
| `train.json`                   | The learner, the actor and `eval_policy.py`                                                           |
| `reward_classifier_train.json` | `lerobot-train` for the reward classifier                                                             |

The robot / workspace values live in both `env.json` and `train.json` (one config class each). `teleoperate.py --write_configs=true` updates both. Every JSON field can be overridden on the command line, e.g. `--dataset.task="Put the cube in the bowl"`.

### 1. Teleoperate, set up the workspace, freeze the step size

```bash
uv run python examples/xarm_hilserl/teleoperate.py \
    --robot.cameras='{wrist: {type: intelrealsense, serial_number_or_name: "134322070906", width: 640, height: 480, fps: 30}}' \
    --home_joints_deg='[62, -19.8, 42, 40.1, 18.5, 54.2, 1.6]' \
    --teleop.use_rotation=true --robot.use_ft_sensor=true --robot.use_admittance=true \
    --end_effector_rot_step=0.06 \
    --reference_clip='{min: [-0.01, -0.01, -0.01, -0.05, -0.05, -0.05], max: [0.01, 0.01, 0.01, 0.05, 0.05, 0.05]}' \
    --write_configs=true \
    --display_data=true

```

The arm first moves to `--home_joints_deg` (here UFACTORY's ready pose; omit it to start where the arm is). A status line shows the live TCP and the min/max visited so far. The arm only moves while you push the puck, inside a ±30 cm box around the start pose (`--box_m`, or an explicit `--end_effector_bounds`). Add `--display_data=true` to see the wrist camera and TCP in Rerun.

To try contacts as they will behave in HIL-SERL, add `--robot.use_admittance=true` and the same `--reference_clip` as in the configs, e.g. `--reference_clip='{min: [-0.01, -0.01, -0.01, -0.05, -0.05, -0.05], max: [0.01, 0.01, 0.01, 0.05, 0.05, 0.05]}'`. The arm then gives way when it is pushed or blocked, and the status line shows the force. The F/T sensor is zeroed at the start pose, so don't touch the gripper while the script starts.

1. **Axis mapping.** Push the puck forward, left and up; the TCP should move +X, +Y, +Z. If not, fix `--teleop.axis_mapping` (e.g. `'[y, -x, z]'`), then set the same `env.teleop.axis_mapping` in `env.json` and `train.json`. Rotation follows the same mapping, so it needs no separate setup. Override it only if you want something else, with `--teleop.rotation_mapping` (`roll` / `pitch` / `yaw` = rotation about the puck x / y / z, e.g. `'[-pitch, roll, yaw]'`).
2. **Step size.** `--end_effector_step_sizes` is the distance moved per step at full deflection (default 1 cm = 10 cm/s at 10 Hz), `--end_effector_rot_step` the rotation (0.06 rad, as hil-serl's `ram_insertion`). Pick values that are precise enough for the task. **Freeze them now**: demos store normalized actions, so changing the step size later changes what the demos mean.
3. **Workspace and home.** The pose the script starts from (`--home_joints_deg`, or where the arm is) becomes the home of every episode.
   - Move the TCP over the whole region the task needs: every corner, highest and lowest point.
   - Press Ctrl+C, wherever the arm is.

On exit the script prints the values it found:

- `env.processor.inverse_kinematics`: `end_effector_bounds` (xyz: the box you visited; |roll|, pitch, yaw: ±`--box_rad` around the start pose, the home of every episode), step sizes, `delta_reference`;
- `env.processor.reset.fixed_reset_joint_positions`: the joints at the start, the home pose of every episode;
- `policy.dataset_stats` for the 19-D state: the visited pose relative to the home, the twist at full deflection, ±`--force_range_n` / `--torque_range_nm`, the gripper.

With `--write_configs=true` it then asks `Write these values into …env.json, …train.json? [y/N]`. Answer `y` only after a good run: a run that ended in an error or did not cover the workspace should not overwrite your configs. Without the flag, nothing is written.

To edit the configs by hand instead, the printed values go here:

| Printed value                                                       | `env.json`                                                                                  | `train.json`           |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------- |
| `end_effector_bounds`, `end_effector_step_sizes`, `delta_reference` | `env.processor.inverse_kinematics`                                                          | same path              |
| `fixed_reset_joint_positions`                                       | `env.processor.reset.fixed_reset_joint_positions` (only this key; keep `reset_time_s` etc.) | same path              |
| `policy.dataset_stats`                                              | not used                                                                                    | `policy.dataset_stats` |

`env.processor.observation` holds observation-processing options, not stats: keep `relative_ee_frame: true` and `add_joint_velocity_to_observation: false` (the state already has the TCP twist). Later steps fill two more fields in both files: the crop in `env.processor.image_preprocessing` (step 4) and the classifier checkpoint in `env.processor.reward_classifier.pretrained_path` (step 5).

Then set the Studio safety boundary a few centimeters outside the box.

`delta_reference` chooses what the delta is applied to:

|                            | `"measured"` (default, as in rail-berkeley/hil-serl) | `"commanded"` (as in LeRobot's SO pipeline)                              |
| -------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| Delta applied to           | The measured TCP                                     | The previous target                                                      |
| After the puck is released | Stops within one step                                | Coasts until it reaches targets it fell behind on                        |
| When blocked / in contact  | Pushes at most one step                              | Target runs ahead, so the push grows (and can trigger a collision error) |
| Speed                      | Slightly below `step × fps`                          | Exactly `step × fps`                                                     |

Compare them with `--delta_reference`. Use the same value for demos and training.

Plain `lerobot-teleoperate` does not work for this pair: it passes the teleop action to the robot unchanged, but the SpaceMouse emits deltas and the xArm takes absolute targets. The same is true of LeRobot's other delta teleops (phone, gamepad), which also come with their own scripts.

### 2. Rehearse in the HIL loop (nothing is saved)

```bash
uv run python -m lerobot.rl.gym_manipulator --config_path examples/xarm_hilserl/configs/env.json
```

This runs exactly the loop used for recording and training, with no policy:

- each episode starts with a reset to `fixed_reset_joint_positions`;
- you rearrange the scene before `reset_time_s` (5 s, counted from the start of the reset, homing included) runs out;
- the robot holds still until you move the SpaceMouse, which then drives it;
- **Enter / Backspace** end the episode;
- after `control_time_s` (20 s) the episode times out.

### 3. Record demonstrations

```bash
uv run python -m lerobot.rl.gym_manipulator --config_path examples/xarm_hilserl/configs/env.json --mode=record
```

For each episode:

1. Wait for the reset.
2. Do the task with the SpaceMouse, starting right away so few idle frames are recorded.
3. Press **Enter** when it's done.

Press **`r`** to discard an attempt that went wrong: like rail-berkeley/hil-serl, keep only successful demos (about 20, `num_episodes_to_record`).

The dataset is written to `~/.cache/huggingface/lerobot/local/xarm7_hilserl_demos_6dof_ft`, with both camera images at full resolution so step 4 can crop them. Demos must contain every camera the policy uses: after adding, removing or moving a camera, record them again.

### 4. Crop and resize the demo images

**This step is always needed, even if you keep the full frame.** The policy takes 128×128 images (`[3, 128, 128]` in `train.json`) but the demos were recorded at 640×480. `crop_dataset_roi` writes the 128×128 copy, `local/xarm7_hilserl_demos_6dof_ft_cropped_resized`, which the learner loads as the offline dataset. **Choosing a smaller box is optional.**

Pick a box per camera:

| Camera                       | What to keep                                                                                                                                                                                                                             | Crop?                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `side` (fixed)               | The table area where the object can be, plus the space the gripper moves through. Leave out chairs, desks and walls: anything that changes between sessions distracts the policy.                                                        | Yes, usually a clear benefit                      |
| `wrist` (moves with the arm) | Everything that can matter during the task: the fingertips and wherever the object can appear in the view. The box is fixed in the image, not on the table, so a tight box around where the object sits in one frame loses it in others. | Often the full frame, or a generous centre square |

Keep boxes roughly square, because they are resized to 128×128. The full 640×480 frame gets squashed slightly, which is fine as long as training uses the same setting, and the configs make sure it does.

**Option A: draw the boxes.** The crop tool needs OpenCV with window support. lerobot installs the windowless build, so add the GUI build for this command only:

```bash
uv run --with "opencv-python>=4.12,<4.14" python -m lerobot.rl.crop_dataset_roi --repo-id local/xarm7_hilserl_demos_6dof_ft
```

It shows the first frame of each camera in turn. Draw a rectangle and press `c` for each. It prints the boxes as `[top, left, height, width]` in camera pixels and writes the 128×128 dataset.

**Option B: type the boxes (no window).** For example, full frame for both cameras:

```bash
echo '{"observation.images.wrist": [0, 0, 480, 640], "observation.images.side": [0, 0, 480, 640]}' > /tmp/crop.json
uv run python -m lerobot.rl.crop_dataset_roi --repo-id local/xarm7_hilserl_demos_6dof_ft --crop-params-path /tmp/crop.json
```

Then put the same boxes into `env.processor.image_preprocessing` of both configs, so live images are processed exactly like the demos:

- `env.json`: set `crop_params_dict` to the boxes (both camera keys) and `resize_size` to `[128, 128]`. Both were `null` (full resolution) for the demos.
- `train.json`: replace the full-frame placeholders `[0, 0, 480, 640]` in `crop_params_dict`. If you kept the full frame, leave them.

Recordings made from now on are already cropped to 128×128.

### 5. Train the reward classifier

The classifier detects success from the camera images (both cameras); it gives the reward and ends successful episodes during training. First record labeled data:

```bash
uv run python -m lerobot.rl.gym_manipulator --config_path examples/xarm_hilserl/configs/env.json \
    --mode=record --dataset.repo_id=local/xarm7_hilserl_classifier --dataset.num_episodes_to_record=15 \
    --env.processor.reset.terminate_on_success=false
```

For each episode (`terminate_on_success=false`, so episodes don't end on success):

1. Do the task with the SpaceMouse.
2. Once it's done, press **Enter** once. From then on every frame of the episode is labeled success, so keep the scene in the success state for a few seconds.
3. End the episode with **Backspace**, or let it time out.

Vary the scene: also record near-misses and failures. Then train:

```bash
uv run lerobot-train --config_path examples/xarm_hilserl/configs/reward_classifier_train.json \
    --output_dir outputs/train/xarm7_reward_classifier
```

To validate it:

1. Set `env.processor.reward_classifier.pretrained_path` in `env.json` to `outputs/train/xarm7_reward_classifier/checkpoints/last/pretrained_model`.
2. Run the dry run.
3. Episodes should end by themselves when the task is done, and not before.

### 6. Train online with human interventions

Before starting, check that `train.json` has:

- the bounds, step sizes, home joints and crop from steps 1 and 4;
- the `policy.dataset_stats` from step 1;
- the classifier path (`outputs/train/xarm7_reward_classifier/checkpoints/last/pretrained_model`).

Then start the learner and the actor in two terminals:

```bash
uv run python -m lerobot.rl.learner --config_path examples/xarm_hilserl/configs/train.json
```

```bash
uv run python -m lerobot.rl.actor --config_path examples/xarm_hilserl/configs/train.json
```

The actor runs the policy on the robot at 10 Hz and streams transitions to the learner over gRPC. The learner trains SAC on a 50/50 mix of:

- online transitions;
- your demos plus every intervention.

`train.json` also selects `mixer: episode_aware` for the **online half** of each
batch. Once the reward classifier marks a completed attempt successful, the
learner indexes its transitions and close/open gripper commands. It reserves
25% of online samples for individual actions from successful attempts, with
twice the episode-selection weight for the shorter half of successes. Of those
selected actions, 25% are drawn from gripper command changes when available.
Separately, it reserves 10% of online samples for a sweep through one complete
successful attempt; a new sweep can start every ten batches after the previous
one finishes. Shorter complete attempts have twice the selection weight, while
longer successes remain eligible. The rest of the online half remains uniformly sampled, so failures
and recovery states remain available to SAC. These are replay weights, not
extra rewards or sequence-model training. Shortness is measured by the end
effector's traveled distance, so compare runs with similar task starts.

To use the original uniform online/offline mix, set `mixer: online_offline`.
The two routes can be adjusted independently with `success_replay_fraction`
(individual-action share), `event_replay_fraction` (gripper event probability
within that share), `trajectory_replay_fraction` (episode-sweep share),
`trajectory_short_weight` (short-vs-long episode selection weight), and
`episode_sweep_interval` (how often a new sweep can start). Set a replay fraction
or the sweep interval to zero to disable its route. The two replay fractions
must sum to at most one. On
resume, saved episodes are indexed again from the replay dataset. Saved data
does not retain whether an action was a human intervention, so that field is
only recorded for new episodes.

It sends updated weights back every `policy_parameters_push_frequency` seconds.

How to intervene:

- Let the policy explore on its own for the first episodes.
- Press **Space** to take over only when it is going wrong, correct with the SpaceMouse, and press **Space** again to hand back.
- Keep interventions short.
- Once it starts succeeding, intervene only for quick nudges, e.g. a grasp.

The intervention rate should fall over time. The learner tracks it as `Intervention rate` in Weights & Biases (set `wandb.enable: true` in `train.json`).

Checkpoints are written to `outputs/train/<date>/<time>_xarm7_hilserl/checkpoints/<step>/pretrained_model` (`last` points to the latest). To resume, run the learner with `--resume=true --config_path=<run>/checkpoints/last/pretrained_model/train_config.json`.

### 7. Run the learned policy

```bash
uv run python examples/xarm_hilserl/eval_policy.py \
    --config_path=examples/xarm_hilserl/configs/train.json \
    --policy.path=outputs/train/<date>/<time>_xarm7_hilserl/checkpoints/last/pretrained_model \
    --eval.n_episodes=10
```

The robot runs the policy for `eval.n_episodes` episodes, with the same resets, classifier and keys as in training. Nothing is trained. Actions are sampled from the policy, as in rail-berkeley/hil-serl's evaluation. At the end the script prints:

- the success rate;
- the mean time to success;
- how many episodes you had to take over in.

## Safety and errors

- **Soft box:** every target is clipped to `end_effector_bounds`. Pushing against an edge is not a failure; the arm just stops there.
- **Hard limits:** the Studio safety boundary, collision detection and self-collision detection stop the arm in the controller. The control box's emergency stop always works.
- **On an xArm error** (collision, boundary, …), commands stop. The episode ends as _truncated_: it's not counted as a failure, and in record mode you can redo it with `r`. The next reset clears the error and moves home.
- **Random reset** (`robot.random_xy_range_m`, default 0 = off) adds a random xy offset to the home pose every episode, as `RANDOM_RESET` in rail-berkeley/hil-serl, whose tasks use 0.01–0.02 m. It makes the policy robust to where it starts; rearranging the scene during resets already provides much of that.

## Tuning and troubleshooting

| Symptom / knob                                                  | What to do                                                                                                                                                                                                                 |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Arm moves after releasing the puck                              | Use `delta_reference: "measured"`. Keep `robot.max_linear_speed_mm_s` above `step size × fps`.                                                                                                                             |
| Step size                                                       | Fixed once demos are recorded. To change it, record the demos again.                                                                                                                                                       |
| `temperature_init` (`algorithm`)                                | 0.01 by default. Higher means more exploration, but interventions then have less effect.                                                                                                                                   |
| `policy_parameters_push_frequency`                              | Seconds between weight updates on the actor; 2 s by default.                                                                                                                                                               |
| Loop slower than 10 Hz                                          | The actor only warns when policy inference is too slow. Robot I/O (TCP and gripper reads, camera) is timed by the summary `teleoperate.py` prints on exit. Check it there first.                                           |
| Replay buffer memory                                            | Buffers live in CPU RAM (`storage_device: cpu`), about 20 GB per 100k transitions. Reduce `online_buffer_capacity` / `offline_buffer_capacity` if needed.                                                                  |
| Reward classifier unreliable                                    | Make sure success is clearly visible in the `side` view, record more varied success and failure examples, and check the crop doesn't cut the object out.                                                                   |
| D405 image dark                                                 | The D405's colour image is darker than the D435i's. Add light to the scene, or set `exposure` / `gain` on the `side` camera config (try values in `realsense-viewer` first). Use the same settings for demos and training. |
| `Failed to open RealSenseCamera(...)`                           | Another program (`realsense-viewer`, a still-running script) has the camera open. Close it. Also check the serial and that it is on USB 3.                                                                                 |
| SpaceMouse not found                                            | Check the udev rule above and that `ls -l /dev/hidraw*` is readable.                                                                                                                                                       |
| Keyboard keys ignored                                           | `pynput` needs an X11 session (not Wayland or SSH without a display).                                                                                                                                                      |
| `pyrealsense2` fails to install                                 | Use Python 3.12 or 3.13.                                                                                                                                                                                                   |
| `src/lerobot/teleoperators/space_mouse/xarm_spacemouse_test.py` | An earlier stand-alone test script, not used here. Move it out of `src/` before opening a PR.                                                                                                                              |
