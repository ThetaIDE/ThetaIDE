# Workspace objectives

`fast_stable` is selected by the CartpoleConvergence experiments. It is an editable objective component loaded through the standard objective API; its descriptor supplies the ordinary configuration controls.

For each training seed, integrate the evaluation mean over the complete fixed transition budget and divide by reward_scale. A stable seed has evaluation mean at least target_reward throughout the final stability_fraction of the budget, with at least min_evaluations actual evaluations in that window. Failed workers count as unstable with zero quality. Curves must cover zero through the budget.

For seed i, quality is normalized full-curve area multiplied by its normalized minimum reward in the final stability window. The tiebreak is 75% mean quality plus 25% mean quality of the worst half of the seeds, with both weights configurable. The trial score is `(number_of_stable_seeds + tiebreak) / (number_of_seeds + 1)`. Thus an additional stable seed always ranks above a speed improvement on fewer stable seeds. Equally reliable configurations are ranked by faster learning and fewer late drops.

Default near-optimal tolerance is 495/500, not literal perfection in every evaluation. Set target_reward to 500 for that stricter definition. Success means stability over the observed finite window, not a guarantee of indefinite stability. Speed means environment transitions, not wall-clock time.

Diagnostics retain per-seed AUC, final reward, late minimum, stable flag, and first observed checkpoint followed by at least min_evaluations consistently near-optimal evaluations through the budget. Compare objective values only within panels with the same seed count and protocol. Evaluate finalists on a common fresh seed panel.

Do not edit objective source or parameters during a running study: objective source is loaded when each trial is scored. Start a new study for this objective; older learning_curve study values are not compatible.
