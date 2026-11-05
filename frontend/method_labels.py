"""Compact display labels; queue and telemetry identifiers retain study identity."""


def parent_trial_label(name: str) -> str:
    """Map a seed source to its trial without collapsing distinct seed feeds."""
    return str(name).split(" · seed ", 1)[0] if " · trial " in str(name) else str(name)


def monitor_method_label(name: str) -> str:
    prefix, separator, trial = str(name).rpartition(" · trial ")
    if separator:
        method = prefix.rsplit(" · ", 1)[0]
        return f"{method} / Trial {trial.replace(' · seed ', ' / Seed ')}"
    return str(name)
