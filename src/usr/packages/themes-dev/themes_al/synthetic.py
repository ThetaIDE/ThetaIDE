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
