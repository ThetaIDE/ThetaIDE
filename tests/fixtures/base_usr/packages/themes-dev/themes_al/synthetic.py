"""Generate artificial visits using the demo's input schema, without clinical data."""
from pathlib import Path
import numpy as np
from .data import PHYSIOLOGY, ACTION


def generate(path, *, visits=40, rows=12, seed=42):
    if visits < 10 or rows < 3:
        raise ValueError('Synthetic input requires at least ten visits and three rows')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(seed)
    names = [f'{f}_mean' for f in PHYSIOLOGY] + [
        'ControlOxygen_max', ACTION, 'VasopressorAdmin_max']
    x = np.zeros((visits, rows + 2, len(names)), dtype=np.float64)
    mask = np.full((visits, rows + 2, 1), -1.0)
    for i in range(visits):
        state = rng.normal(0, 0.5, (rows, len(PHYSIOLOGY)))
        state[rows // 2:] += (1 if i % 2 else -1) * 1.5
        x[i, 2:, :len(PHYSIOLOGY)] = state
        probability = 1 / (1 + np.exp(-2 * np.r_[state[0, 0], state[:-1, 0]]))
        x[i, 2:, names.index(ACTION)] = rng.binomial(1, probability)
        mask[i, 2:] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=x, mask=mask, y=np.arange(visits) % 2,
                        vids=np.array([f'synthetic-{i}' for i in range(visits)]),
                        feature_names_x=np.array(names), synthetic=np.array(True))
    return path


def generate_time_csv(path, *, visits=30, rows=18, seed=7):
    """Artificial irregular-time visits in the downloaded THEMES CSV schema.

    Physiology drifts between two regimes halfway through each visit, timestamps
    are strictly increasing minutes at irregular gaps, and the binary policy
    target `a_2` follows the first feature. `Action` and the other flags are zero
    and `Shock` is empty: they are never model inputs.
    """
    import pandas as pd
    if visits < 10 or rows < 3:
        raise ValueError('Synthetic input requires at least ten visits and three rows')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(visits):
        state = rng.normal(-2, 0.3, (rows, len(PHYSIOLOGY)))
        state[rows // 2:] += (1 if i % 2 else -1) * 0.75
        frame = pd.DataFrame(state, columns=PHYSIOLOGY)
        frame.insert(len(PHYSIOLOGY), 'VisitIdentifier', f'synthetic_{i:03d}')
        gaps = rng.choice([5, 10, 30, 90], size=rows - 1)
        frame.insert(len(PHYSIOLOGY) + 1, 'MinutesFromArrival', np.r_[0, np.cumsum(gaps)].astype(int))
        probability = 1 / (1 + np.exp(-4 * (state[:, 0] - state[:, 0].mean())))
        frame['a_0'], frame['a_1'] = 0, 0
        frame['a_2'] = rng.binomial(1, probability)
        frame['a_3'], frame['Action'], frame['Shock'] = 0, 0, np.nan
        frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def write_resources(directory):
    """Write the package's local demonstration inputs, which Git does not track."""
    directory = Path(directory)
    for name, write in (('synthetic.npz', generate), ('synthetic_time.csv', generate_time_csv)):
        if not (directory / name).exists():
            write(directory / name)
    return directory
