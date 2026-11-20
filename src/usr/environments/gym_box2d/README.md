# Gymnasium Box2D (local)

Select `env/gym_box2d/lunarlander` for discrete `LunarLander-v3` (eight native
observation features, four actions). This editable local component currently
supports Lunar Lander; continuous landers, walkers and racing are outside its scope.

Install dependencies in the training interpreter:

```bash
venv/bin/python -m pip install 'gymnasium[box2d]>=1.0,<1.1'
```

The provider exposes Theta's vector reset/step interface. The SB3 runner constructs
the same standard Gymnasium ID directly. No custom dynamics or reward shaping is
used. Engine action order is `noop`, `fire_left`, `fire_main`, `fire_right`.
Use `observation_scaling: none` with the supplied symbolic rules. The default
profile uses standard gravity and wind disabled; custom physics require runner
support as well as provider support and are not exposed by this component.

Experiments are in `src/usr/config/experiment/BlendRL/LunarLander/`. Start with
`Pilot`; see `docs/guides/LUNARLANDER_BLENDRL.md` for the research sequence,
budgets and tuning/held-out distinctions.

```bash
venv/bin/python -m pytest -q src/usr/environments/gym_box2d/tests
venv/bin/python src/usr/environments/gym_box2d/evaluate_rules.py \
  --out src/usr/results/validation/lunarlander_rules.json
venv/bin/python run_pipeline.py BlendRL/LunarLander/Smoke
```

Raw-controller evaluation draws and records one seed unless `--seed` is supplied.
The two controllers use matched episode seeds. It measures deterministic rules,
separately from the randomly initialized, learned NSFR clause weights in `Logic`.
