"""Guard the original-source integration. Never replace absent source with a demo."""
from pathlib import Path
import hashlib
import json


MISSING = {
    'student/EDMStudent': 'EDM objective, optimization and persistent sampling implementation',
    'preprocessing/EHR_demos_generator': 'TICCSplitTrajs, REGTICCSplitTrajs and PartitionTICCSubTrajs',
    '__head__': 'shared imports and original parameter context',
    'original experiment parameters': 'para_config.npy, convergence thresholds and source split provenance',
}


def verify_mirror():
    root = Path(__file__).parent / 'reference/original'
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Original THEMES source checksum mismatch: {name}')
    return manifest


def require_original_source():
    verify_mirror()
    raise ValueError(
        'Original THEMES execution is unavailable: the previous runner was demo-derived and has been disabled. '
        'Missing upstream definitions: ' + '; '.join(f'{key} ({why})' for key, why in MISSING.items()) + '. '
        'The model must be selected in model configuration, not constructed by the algorithm. '
        'NV_MLIRL.py is present, but the source-faithful outer reward-feedback schedule is not integrated. '
        'No demo fallback or configuration rewrite will be performed. See docs/guides/THEMES_SOURCE_AUDIT.md.')
