"""Versioned inference bundles. Never fit scalers/models or resume training here."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .core import EMEDM, MultiSeriesTICC, Segment, make_segments

VERSION = 1
FILES = ('policies.pt', 'preprocessing_and_stages.npz', 'grouped_segments.npz', 'segments.csv')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def seal(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['artifact_sha256'] = {name: sha256(output / name) for name in FILES}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


class InferenceBundle:
    def __init__(self, output, *, resume=False, model_resolver=None):
        if resume:
            raise ValueError('Exact training resume is unsupported: optimizer, SGLD, RNG and EM state are not persisted')
        self.output = Path(output)
        self.manifest = json.loads((self.output / 'manifest.json').read_text())
        if self.manifest.get('artifact_version') != VERSION:
            raise ValueError('Unsupported THEMES artifact version')
        hashes = self.manifest.get('artifact_sha256', {})
        for name in FILES:
            if name not in hashes or sha256(self.output / name) != hashes[name]:
                raise ValueError(f'Artifact checksum mismatch: {name}')
        with np.load(self.output / 'preprocessing_and_stages.npz', allow_pickle=False) as fitted:
            self.fitted = {name: fitted[name] for name in fitted.files}
        if 'model_bundle' in self.manifest:
            from .models import instantiate_stage
            self.stage = instantiate_stage(self.manifest['model_bundle']['stage'],
                self.manifest['parameters']['seed'], model_resolver)
        else:
            # Explicit historical version-1 stage reader.
            self.stage = MultiSeriesTICC(switch_penalty=self.manifest['parameters']['switch_penalty'])
        self.stage.means = self.fitted['stage_means']
        self.stage.precisions = self.fitted['stage_precisions']
        checkpoint = torch.load(self.output / 'policies.pt', map_location='cpu', weights_only=True)
        from .models import Policy, instantiate_policy
        version = checkpoint.get('artifact_version')
        if version == 1:
            if checkpoint.get('beta') != 0.5:
                raise ValueError('Unsupported legacy demo logit scale')
            # Explicit historical reader only; no training uses this architecture.
            factory = lambda dimension: Policy(dimension, checkpoint['width'])
        elif version == 2:
            specs = checkpoint['model_specs']
            if len(specs) != len(checkpoint['policies']):
                raise ValueError('Policy specifications and weights are misaligned')
            factory = lambda dimension: instantiate_policy(specs[0], dimension, model_resolver)
        else:
            raise ValueError('Unsupported policy checkpoint version')
        self.mixture = EMEDM(policy_factory=factory)
        self.mixture.models = []
        for index, weights in enumerate(checkpoint['policies']):
            model = (instantiate_policy(specs[index], checkpoint['dimension'], model_resolver)
                     if version == 2 else factory(checkpoint['dimension']))
            model.load_state_dict(weights)
            model.eval()
            self.mixture.models.append(model)
        self.mixture.priors = np.array(checkpoint['priors'])
        if (not len(self.mixture.models) or len(self.mixture.priors) != len(self.mixture.models)
                or not np.isfinite(self.mixture.priors).all()
                or (self.mixture.priors < 0).any()
                or not np.isclose(self.mixture.priors.sum(), 1)):
            raise ValueError('Invalid mixture priors')

    def transform(self, episodes):
        """Use persisted train-fitted preprocessing on fresh visit arrays."""
        dimension = len(self.fitted['physiology_mean'])
        if not episodes or any(x.ndim != 2 or x.shape[1] != dimension or not len(x)
                               or not np.isfinite(x).all() for x in episodes):
            raise ValueError('Expected nonempty finite visit arrays with matching feature schema')
        offsets = np.r_[0, np.cumsum([len(x) for x in episodes])]
        stages, policies, validity = [], [], []
        for x in episodes:
            stages.append(np.clip((x - self.fitted['physiology_mean']) /
                                  self.fitted['physiology_scale'], -5, 5))
            low = np.array([x[max(0, t - 2):t + 1].min(axis=0) for t in range(len(x))])
            high = np.array([x[max(0, t - 2):t + 1].max(axis=0) for t in range(len(x))])
            aggregate = np.concatenate([x, high, low], axis=1)
            lagged = np.vstack([aggregate[:1], aggregate[:-1]])
            policies.append(np.clip(2 * (lagged - self.fitted['policy_lower']) /
                                    self.fitted['policy_span'] - 1, -1, 1).astype(np.float32))
            validity.append(np.arange(len(x)) > 0)
        return stages, np.concatenate(policies), np.concatenate(validity), offsets

    def predict(self, episodes, actions):
        if len(episodes) != len(actions) or any(len(x) != len(a) or np.asarray(a).ndim != 1
                or not np.isin(a, [0, 1]).all() for x, a in zip(episodes, actions)):
            raise ValueError('Expected one aligned binary action array per visit')
        stage_input, policy_input, valid, offsets = self.transform(episodes)
        stages = self.stage.predict(stage_input)
        segments = make_segments(stages, offsets, ['score'] * len(episodes))
        probabilities, evidence = self.mixture.predict(policy_input, np.concatenate(actions), valid, segments)
        return dict(stages=stages, segments=segments, probabilities=probabilities, evidence=evidence,
                    policy_states=policy_input, valid=valid, offsets=offsets)


def verify(output, *, source=True):
    """Reload in a fresh process; verify all memberships and optional source lineage."""
    output = Path(output)
    if json.loads((output / 'manifest.json').read_text()).get('artifact_version') == 2:
        from .source_artifacts import verify_source
        return verify_source(output,source=source)
    bundle = InferenceBundle(output)
    with np.load(output / 'grouped_segments.npz', allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    offsets = data['visit_offsets']
    episodes = [data['observations'][a:b].astype(np.float64) for a, b in zip(offsets[:-1], offsets[1:])]
    if source:
        from .data import load_data
        path = Path(bundle.manifest['dataset'])
        if sha256(path) != bundle.manifest['dataset_sha256']:
            raise ValueError('Source dataset checksum mismatch')
        if bundle.manifest.get('source_format') == 'csv':
            from .csv_data import load_csv
            original = load_csv(path, bundle.manifest['parameters']['patients'], bundle.manifest['parameters']['seed'],
                                **(bundle.manifest.get('source_schema') or {}))
        else:
            original = load_data(path, bundle.manifest['parameters']['patients'], bundle.manifest['parameters']['seed'])
        np.testing.assert_array_equal(original['vids'], data['visit_ids'])
        np.testing.assert_array_equal(original['source_indices'], data['source_patient_indices'])
        np.testing.assert_array_equal(np.concatenate(original['slots']), data['source_slots'])
        np.testing.assert_array_equal(np.concatenate(original['actions']), data['actions'])
        np.testing.assert_array_equal(np.concatenate(original['interventions']), data['intervention_flags'])
        if original.get('timestamps') is not None:
            np.testing.assert_array_equal(np.concatenate(original['timestamps']), data['minutes_from_arrival'])
            np.testing.assert_array_equal(original['observation_feature_names'], data['observation_feature_names'])
        np.testing.assert_allclose(np.concatenate(original['observations']), data['observations'], rtol=1e-6, atol=1e-6)
        episodes = original['observations']  # Preserve the source's float64 precision at decision boundaries.
    actions = [data['actions'][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    result = bundle.predict(episodes, actions)
    np.testing.assert_array_equal(np.concatenate(result['stages']), data['physiological_stages'])
    np.testing.assert_allclose(result['policy_states'], data['policy_observations'], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(result['valid'], data['policy_scoring_mask'])
    segments = [Segment(int(p), int(a), int(b), int(data['physiological_stages'][a]), str(data['visit_split'][p]))
                for p, a, b in zip(data['segment_visit_index'], data['segment_start'], data['segment_stop'])]
    # Use exact stored float32 policy states when reproducing the saved likelihood.
    probs, evidence = bundle.mixture.predict(data['policy_observations'], data['actions'],
                                            data['policy_scoring_mask'], segments)
    np.testing.assert_allclose(probs, data['segment_policy_probabilities'], atol=1e-10, rtol=1e-10)
    coverage = np.zeros(len(data['actions']), dtype=int)
    for i, s in enumerate(segments):
        if not offsets[s.patient] <= s.start < s.stop <= offsets[s.patient + 1]:
            raise ValueError('Segment crosses visit boundary')
        np.testing.assert_array_equal(data['segment_ids'][s.start:s.stop], i)
        coverage[s.start:s.stop] += 1
    np.testing.assert_array_equal(coverage, 1)
    test = np.array([s.split == 'test' for s in segments])
    test_rows = np.repeat(data['visit_split'] == 'test', np.diff(offsets)) & data['policy_scoring_mask']
    nll = float(-evidence[test].sum() / test_rows.sum())
    np.testing.assert_allclose(nll, bundle.manifest['metrics']['heldout_mixture_nll_per_scored_step'], atol=1e-10)
    result = dict(status='passed', artifact_version=VERSION, source_lineage_checked=source,
                  visits=len(episodes), segments=len(segments), scored_rows=int(data['policy_scoring_mask'].sum()),
                  heldout_mixture_nll_per_scored_step=nll)
    (output / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
