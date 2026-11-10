"""Individual outputs transported by this runner; no SB3/torch imports."""


def describe(config, context):
    agent = config
    algorithm = str(agent.get("algorithm", "PPO")).rsplit("/", 1)[-1].upper()
    outputs = []

    def add(name, group):
        outputs.append({"name": name, "label": name.rsplit("/", 1)[-1].replace("_", " "), "group": group})

    for name in ("episode_reward", "episode_length", "reward", "length"):
        add("train/" + name, "Training episodes")
    for name in ("reward", "reward_std"):
        add("eval/" + name, "Evaluation")
    for mode in ("greedy", "sampled"):
        for suffix in ("reward", "reward_std"):
            add(f"eval/{mode}_{suffix}", "Evaluation")
    for name in ("fps", "time_elapsed"):
        add("time/" + name, "Timing")
    losses = {
        "PPO": ["entropy_loss", "policy_gradient_loss", "value_loss", "loss"],
        "A2C": ["entropy_loss", "policy_loss", "value_loss"],
        "DQN": ["loss"],
        "SAC": ["actor_loss", "critic_loss"],
    }.get(algorithm, [])
    if algorithm == "SAC" and str(agent.get("ent_coef", "auto")).startswith("auto"):
        losses.append("ent_coef_loss")
    for name in losses:
        add("losses/train/" + name, "Optimization losses")
    diagnostics = ["learning_rate", "n_updates"]
    if algorithm == "PPO":
        diagnostics += ["approx_kl", "clip_fraction", "explained_variance", "clip_range"]
        if agent.get("clip_range_vf") is not None:
            diagnostics.append("clip_range_vf")
    elif algorithm == "A2C":
        diagnostics.append("explained_variance")
    elif algorithm == "SAC":
        diagnostics.append("ent_coef")
    # Selected Theta PPO/A2C policies have no log_std. Native continuous policies
    # and SAC's SDE policy can additionally emit their action standard deviation.
    if (context.get("model") or {}).get("sb3_native_policy") is True and algorithm in ("PPO", "A2C"):
        diagnostics.append("std")
    if algorithm == "SAC" and agent.get("use_sde"):
        diagnostics.append("std")
    for name in diagnostics:
        add("diagnostics/" + name, "Optimizer diagnostics")
    return outputs
