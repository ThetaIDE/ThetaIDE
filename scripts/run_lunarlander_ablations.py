"""Launch the prepared feature sweeps, then prepare held-out winner configs."""
from pathlib import Path
import argparse
import copy
import json
import os
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'src/usr/results/sweep_logs/lunarlander_bets_150k_20261119'
RECIPES = ROOT / 'src/usr/config/experiment/BlendRL/LunarLander/Bets'


def save(directory, value):
    target = directory / 'state.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(target)


def run(directory, recipes, selector):
    import optuna
    # Keep promotion templates private to this launch; later editor changes do
    # not alter which methods/configs this coordinator promotes.
    template = yaml.safe_load((recipes / 'Held-out Ablations Template.yaml').read_text())
    config = yaml.safe_load((recipes / 'Feature Sweeps 150k.yaml').read_text())
    options = config['tuning']['options']['optuna']
    (directory / 'promotion_template.yaml').write_text(yaml.safe_dump(template, sort_keys=False))
    state = {'status': 'running', 'pid': os.getpid(), 'selector': selector,
             'automatic_held_out_training': False, 'methods': list(config['methods'])}
    provenance_path = directory / 'base_provenance.json'
    baseline = json.loads(provenance_path.read_text()).get('matched_baseline') if provenance_path.is_file() else None
    if baseline:
        state['matched_baseline'] = baseline
    save(directory, state)
    try:
        with (directory / 'Feature Sweeps.log').open('w') as log:
            child = subprocess.Popen([sys.executable, '-u', 'run_pipeline.py', selector], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT)
            state['pipeline_pid'] = child.pid
            save(directory, state)
            code = child.wait()
        if code:
            raise RuntimeError(f'Feature sweep launch exited {code}; see Feature Sweeps.log')
        selected = {}
        for name, method in config['methods'].items():
            study_name = options['study_name'].format(method=name)
            study = optuna.load_study(study_name=study_name,
                                     storage='sqlite:///' + str(ROOT / 'src/usr/results/optuna/optuna.db'))
            if study.user_attrs.get('theta_last_stop_reason') != 'trial_limit':
                raise RuntimeError(f'{study_name} ended before reaching its trial limit')
            trial = study.best_trial
            selected[name] = {'study': study_name, 'trial': trial.number, 'value': trial.value, 'params': trial.params}
            if baseline:
                selected[name]['delta_from_standard_blendrl'] = trial.value - baseline['value']
            for key, value in trial.params.items():
                if key not in method['tune']:
                    raise ValueError(f'Unexpected promoted parameter {name}/{key}')
                target = template['methods'][name]
                parts = key.split('.')
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = copy.deepcopy(value)
        template['description'] = (
            'Held-out full-controller BlendRL ablations. Variant parameters come from the completed '
            'feature sweeps. Fresh matched training and evaluation seeds are drawn at launch.'
        )
        target = recipes / 'Tuned Held-out Ablations.yaml'
        target.write_text('# @package _global_\n' + yaml.safe_dump(template, sort_keys=False))
        held_out = selector.rsplit('/', 1)[0] + '/Tuned Held-out Ablations'
        subprocess.run([sys.executable, '-u', 'run_pipeline.py', held_out, 'dry_run=true'],
                       cwd=ROOT, check=True)
        state.update(status='completed', winners=selected, held_out_recipe=str(target))
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(directory, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--batch', help='Batch folder under LunarLander/Bets')
    args = parser.parse_args()
    if args.batch and (Path(args.batch).name != args.batch or args.batch in ('.', '..')):
        parser.error('--batch must be a single folder name')
    recipes = RECIPES / args.batch if args.batch else RECIPES
    directory = DIRECTORY.parent / 'lunarlander_bets' / args.batch if args.batch else DIRECTORY
    selector = recipes.relative_to(ROOT / 'src/usr/config/experiment').as_posix() + '/Feature Sweeps 150k'
    os.chdir(ROOT)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.thetaide/matplotlib'))
    if args.start:
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with (directory / 'state.json').open('x') as state_file:
                json.dump({'status': 'starting', 'selector': selector}, state_file)
        except FileExistsError:
            parser.error('This feature batch already has a state record; inspect it before starting another launch')
        command = [sys.executable, '-u', __file__]
        if args.batch:
            command.extend(['--batch', args.batch])
        with (directory / 'coordinator.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'directory': str(directory)}, indent=2))
    else:
        run(directory, recipes, selector)
