# Training (SAC + DQN gripper, as rail-berkeley/hil-serl)

Learner: `lerobot.rl.learner` with `configs/train.json`. Code: `src/lerobot/rl/algorithms/sac/sac_algorithm.py`, `src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py`. Reference: `hil-serl/` (`serl_launcher/agents/continuous/sac_hybrid_single.py`).

## Data

**Transition** $(s, a, r, r_g, s', d)$

| Symbol | Meaning |
|---|---|
| $s = (I_{\text{wrist}}, I_{\text{side}}, I_{\text{front}}, x)$ | 3 images $\in \mathbb{R}^{3\times128\times128}$ (ImageNet-normalized) + raw state $x \in \mathbb{R}^{19}$ (not normalized) |
| $x$ | pose relative to reset (6), EE-frame twist (6), tool-frame force / torque (6), gripper (1) |
| $a = (a_c, a_g)$ | $a_c \in [-1,1]^6$ EE-frame delta pose, $a_g \in \{0,1,2\}$ = close / stay / open |
| $r$ | $\mathbb{1}[\text{classifier}(s') > 0.5]$ |
| $r_g$ | $-0.02$ if $a_g$ toggles the gripper, else $0$ |
| $d$ | $1$ on success (episode ends), else $0$ |

**Batch** $B$: 256 transitions, half from the online buffer, half from the offline buffer (demos + interventions). Images of $s$ and $s'$ are randomly shifted by up to 4 px.

## Networks

**Encoder** $\phi$ (one copy, shared by all networks), and a target copy $\bar\phi$:

$$z_k = \tanh\big(\mathrm{LN}(W_k\,\mathrm{Drop}(\mathrm{SpatialEmb}_k(\mathrm{ResNet10}(I_k))))\big) \in \mathbb{R}^{256}, \qquad z_x = \tanh(\mathrm{LN}(W_x x)) \in \mathbb{R}^{64}$$

$$\phi(s) = [z_{\text{wrist}}, z_{\text{side}}, z_{\text{front}}, z_x] \in \mathbb{R}^{832}$$

ResNet10 is frozen. $\mathrm{Drop}$ (p = 0.1) is applied on the actor side only.

| Network | Input | Output |
|---|---|---|
| Actor $\pi$ | $\phi(s)$, image part detached | $\mu, \sigma \in \mathbb{R}^6$, $\sigma \in [10^{-5}, 5]$; $a_c = \tanh(u),\ u \sim \mathcal{N}(\mu, \sigma^2)$ |
| Critics $Q_{1,2}$ | $[\phi(s), a_c]$ | $\mathbb{R}$ |
| Gripper critic $Q^g$ | $\phi(s)$ | $\mathbb{R}^3$, one value per $a_g$ |
| Targets $\bar Q_{1,2}, \bar Q^g$ | same, through $\bar\phi$ | same |
| Temperature | — | $\alpha = \mathrm{softplus}(p)$, $\alpha_0 = 0.01$ |

All MLPs are $[256, 256]$ with LayerNorm and $\tanh$.

$$\log\pi(a_c|s) = \log\mathcal{N}(u;\mu,\sigma) - \textstyle\sum_{i=1}^{6}\log(1-\tanh^2 u_i)$$

## Losses ($\gamma = 0.98$, $a' \sim \pi(\cdot|s')$, $\tilde a \sim \pi(\cdot|s)$)

$$y = r + \gamma(1-d)\,\min_{i}\bar Q_i(s',a'), \qquad \mathcal{L}_Q = \sum_{i=1}^{2}\mathbb{E}_B\big[(Q_i(s,a_c) - y)^2\big]$$

$$y^g = r + r_g + \gamma(1-d)\,\bar Q^g(s')_{g^*},\quad g^* = \arg\max_g Q^g(s')_g, \qquad \mathcal{L}_g = \mathbb{E}_B\big[(Q^g(s)_{a_g} - y^g)^2\big]$$

$$\mathcal{L}_\pi = \mathbb{E}_B\Big[\alpha\log\pi(\tilde a|s) - \tfrac{1}{2}\textstyle\sum_i Q_i(s,\tilde a)\Big] \quad (\text{critic encoder detached})$$

$$\mathcal{L}_\alpha = \mathbb{E}_B\big[-\alpha\,(\log\pi(a'|s') + \bar{\mathcal{H}})\big], \qquad \bar{\mathcal{H}} = -7/2$$

**Which loss trains what**

| Loss | Trains |
|---|---|
| $\mathcal{L}_Q$ | $Q_{1,2}$, all of $\phi$ (except ResNet10) |
| $\mathcal{L}_g$ | $Q^g$, all of $\phi$ (except ResNet10) |
| $\mathcal{L}_\pi$ | $\pi$, state layer $W_x$ of $\phi$ |
| $\mathcal{L}_\alpha$ | $p$ |

## Update

One learner step uses 2 batches (`utd_ratio` 2):

1. $\{\mathcal{L}_Q, \mathcal{L}_g\}$, then target update.
2. $\{\mathcal{L}_Q, \mathcal{L}_g, \mathcal{L}_\pi, \mathcal{L}_\alpha\}$, then target update.

Within a step, all gradients are taken at the same parameters; each loss has its own Adam (lr $3\times10^{-4}$, no clipping), and parameters shared between losses get the sum of the updates.

$$\bar\theta \leftarrow 0.005\,\theta + 0.995\,\bar\theta \qquad (\bar Q_{1,2}, \bar Q^g, \bar\phi)$$

The learner steps as fast as the GPU allows, independent of the 10 Hz robot. It sends $\pi$ and $Q^g$ to the actor every 50 steps, logs every 10 steps (wandb) and checkpoints every 2000 steps. Training starts once the online buffer holds 100 transitions.

## Output

$$a = \big[\tanh(u),\ \arg\max_g Q^g(s)_g\big], \qquad u \sim \mathcal{N}(\mu(s), \sigma(s)^2)$$

The env scales $a_c$ by 0.01 m (translation) and 0.06 rad (rotation) per step.

---

## Appendix: changes to follow hil-serl

All changes need a fresh run: old checkpoints have a different architecture and were trained on base-frame force. Options that are new in the code default to the old lerobot behavior; `train.json` turns them on.

### Bug fixes

| Change | Before | After | Pros | Cons | Consequence |
|---|---|---|---|---|---|
| Gripper penalty in the gripper-critic target (`sac_algorithm.py`, main batch `include_complementary_info=True`) | $y^g = r + \gamma(1-d)\bar Q^g$; with `utd_ratio` 1 the $-0.02$ never reached training | $y^g = r + r_g + \gamma(1-d)\bar Q^g$ | Breaks ties between close / stay / open in favor of "stay" | None | Less gripper chatter; the gripper toggles only when it pays off by more than 0.02 |
| Force / torque frame (`xarm.py`, `config_xarm.py`; `ft_sensor_frame: "tool"` now means unrotated) | Rotated into the base frame (comment wrongly said hil-serl uses `O_F_ext_hat_K`) | Tool frame, as reported, like hil-serl's `K_F_ext_hat_K` | Same contact gives the same reading at any gripper yaw; force shares a frame with twist and actions | Old data and checkpoints are in the base frame | Demos converted with `convert_ft_frame.py` to `local/xarm7_hilserl_demos_6dof_ft_toolft` (`dataset.repo_id`); old checkpoints need `"base"` |
| Target gripper critic built from the policy's network config | Built from the algorithm's copy of the config | Same config as the online gripper critic | Online and target networks cannot silently differ (activations have no weights to catch it) | None | No change with matching configs; needed once activations became configurable |

### Hyperparameters (config only)

| Change | Before | After | Pros | Cons | Consequence |
|---|---|---|---|---|---|
| Critic updates per step (`utd_ratio`) | 1 | 2 (hil-serl `cta_ratio`) | Critic stays ahead of the actor; better sample efficiency | ~2× critic compute per step | Lower logged Hz; more updates per robot step overall |
| State normalization (`dataset_stats` of `observation.state`) | Min-max with workspace stats (force ±30 N → ±1) | Raw ($\pm1$ stats = identity) | Same inputs as hil-serl; no hand-tuned stats to get wrong | Inputs on very different scales (pose ~0.1 m, force up to ~20 N); force dominates the state layer | The state layer must learn the scales; LayerNorm makes large forces look alike in magnitude |
| Entropy term in the critic target (`use_backup_entropy`) | On: $y$ includes $-\alpha\log\pi(a'|s')$ | Off | Simpler, less noisy target; hil-serl default | Q no longer includes future entropy bonuses | With $\alpha \approx 0.01$ the difference is small |
| Ensemble reduction in the actor loss (`actor_q_reduction`) | $\min_i Q_i$ | $\text{mean}_i Q_i$ | Less pessimistic, smoother actor gradient | More exposed to critic overestimation | Bolder actions early on |
| Activations (actor, critics, gripper critic) | SiLU | Tanh | Bounded features; matches hil-serl | Saturates for large inputs | Slightly different learning dynamics |
| Gripper critic last hidden layer (`activate_final`) | LayerNorm + activation | Plain linear | Matches hil-serl | — | Marginal |
| State layer size (`state_encoder_hidden_dim`, now used by the encoder) | 256 | 64 | State takes a smaller share of $\phi$ (64 of 832), as hil-serl | Less capacity for force / pose | Images weigh more in the features |
| Policy std upper limit (`std_max`) | 10 | 5 | Less extreme exploration | Less exploration | Fewer saturated $\pm1$ actions early |
| Gradient clipping (`grad_clip_norm`) | 40 | None | Matches hil-serl; no clipping bias | No protection against gradient spikes | Watch for loss spikes in wandb |
| Weight push to the actor (`policy_parameters_push_steps`, new) | Every 2 s | Every 50 optimization steps | Push rate tied to learning progress, as hil-serl | Push interval in seconds now depends on learner speed | At ~20 Hz, one push every ~2.5 s |
| Buffer capacity (online and offline) | 100k | 200k | Keeps older data longer | More RAM | Buffers fill after ~5.5 h at 10 Hz |
| Discount $\gamma$ | 0.97 | 0.98 (hil-serl `usb_pickup_insertion`) | Longer horizon (~50 steps) | Slower value propagation, higher variance | Values credit success from further back |
| Checkpoint / log / wandb | 5000 / 500 / off | 2000 / 10 / on | Losses visible; finer checkpoints | More disk and logging overhead | Needs `wandb login` |

### Algorithm (new options in code)

| Change | Before | After | Pros | Cons | Consequence |
|---|---|---|---|---|---|
| Actor trains the state layer (`actor_trains_state_encoder`) | Actor optimizer excluded the whole shared encoder | Includes $W_x$ (image part still detached) | State features shaped for the actor too, as hil-serl | Actor and critics pull the shared state layer in different directions | $W_x$ gets three Adam updates per step (critic, gripper critic, actor) |
| Actor loss detaches the critic's encoder (always on) | Actor-loss gradient also flowed into $W_x$ through the critic | Only through the actor's path | Matches hil-serl's gradient exactly | None | No effect unless the actor trains the state layer |
| Target encoder (`use_target_encoder`) | Targets used the online encoder | Separate Polyak copy $\bar\phi$, saved in checkpoints | More stable targets: encoder changes do not move $y$ immediately | Extra encoder copy in memory | Targets lag the encoder like the critic heads |
| Init (`hil_serl_init`) | PyTorch defaults; orthogonal heads; actor output uniform ±0.05 | Flax defaults: Xavier uniform (MLPs, heads, state layer), truncated LeCun normal (image layers), zero biases | Same starting point as hil-serl; spatial-embedding std 8× smaller than before | Initial Q and actions less damped than with ±0.05 heads | Different early-training behavior |
| Temperature (`temperature_parameterization`) | $\alpha = e^{p}$ | $\alpha = \mathrm{softplus}(p)$ | Matches hil-serl | Near-linear for large $\alpha$ | Nearly identical at $\alpha \approx 0.01$ |
| Temperature entropy at $s'$ (`temperature_uses_next_observations`) | Entropy at $s$ | Entropy at $s'$ | Matches hil-serl | — | Same on average: $s$ and $s'$ come from the same buffer |
| Simultaneous updates (`simultaneous_updates`) | Critic, gripper critic, actor, temperature stepped in turn | All gradients at the same parameters, Adam updates summed | Matches hil-serl; no loss sees another's update from the same step | Slightly more memory (graphs kept until all gradients are taken) | Verified against a by-hand step to 1.6e-7 |
| Critic image dropout (`critic_image_dropout`) | Dropout in critics and actor | Actor only | Deterministic critic targets and predictions, as hil-serl | Less regularization of the critics | Lower critic-loss noise |

### Not changed

| Item | Why |
|---|---|
| Critic loss summed over the 2 critics (hil-serl averages) | A constant factor; Adam updates are identical without clipping |
| Episode length (900 vs hil-serl 120 steps), classifier threshold (0.5 vs 0.7 + state check), random reset (none vs ±1 cm / ±0.1 rad) | Task-specific |
| Action scale (0.01 m / 0.06 rad vs 0.015 / 0.1) | Changing it invalidates the demos |
