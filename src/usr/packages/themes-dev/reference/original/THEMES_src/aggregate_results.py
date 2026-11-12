"""
Aggregates per-repeat CSV files written by SLURM job array runs into a single summary CSV.
Supports: THEMES, EM_EDM, EDM, IQLearn, PWDICE, CHALIDE, SOIL, THEMES_CD, CRUX

Usage:
    python aggregate_results.py                                                        # THEMES, CCHS, unweighted, expert-only
    EHR_ENV=MIMIC USE_HALIDE_W=1 python aggregate_results.py
    MODEL=EM_EDM EHR_ENV=MIMIC python aggregate_results.py
    MODEL=EDM EHR_ENV=MIMIC python aggregate_results.py
    CLEANTRAIN=1 EHR_ENV=MIMIC python aggregate_results.py
    MODEL=EM_EDM CLEANTRAIN=1 EHR_ENV=MIMIC USE_HALIDE_W=1 python aggregate_results.py
    DATA_MODE=ei EHR_ENV=MIMIC python aggregate_results.py                              # expert + suboptimal (E+I)
    DATA_MODE=all EHR_ENV=MIMIC python aggregate_results.py                             # all 4 quadrants
    MODEL=THEMES CLEANTRAIN=1 DATA_MODE=e USE_HALIDE_W=0 EHR_ENV=MIMIC python aggregate_results.py
    MODEL=IQLearn DATA_MODE=e EHR_ENV=MIMIC python aggregate_results.py
    MODEL=PWDICE DATA_MODE=ei EHR_ENV=MIMIC python aggregate_results.py
    MODEL=CHALIDE DATA_MODE=e USE_HALIDE_W=1 USE_DICE=1 EHR_ENV=CCHS python aggregate_results.py
    MODEL=CHALIDE DATA_MODE=e USE_HALIDE_W=1 USE_DICE=0 EHR_ENV=CCHS python aggregate_results.py
    MODEL=SOIL DATA_MODE=all USE_HALIDE_W=0 EHR_ENV=MIMIC RUN_TAG=v1 python aggregate_results.py
    MODEL=SOIL DATA_MODE=all USE_HALIDE_W=1 EHR_ENV=MIMIC RUN_TAG=v1 python aggregate_results.py
    MODEL=THEMES_CD DATA_MODE=all USE_HALIDE_W=0 EHR_ENV=MIMIC RUN_TAG=v1 python aggregate_results.py
    MODEL=THEMES_CD DATA_MODE=all USE_HALIDE_W=1 EHR_ENV=MIMIC RUN_TAG=v1 python aggregate_results.py
    ITER_SEL=last MODEL=THEMES CLEANTRAIN=1 DATA_MODE=e USE_HALIDE_W=0 EHR_ENV=MIMIC python aggregate_results.py
    MODEL=EM_EDM HALIDE_VARIANT=a2_t37 DATA_MODE=ei USE_HALIDE_W=1 EHR_ENV=Mayo RUN_TAG=bal1 python aggregate_results.py
    MODEL=EM_EDM HALIDE_VARIANT=a2_t37 DATA_MODE=ei USE_HALIDE_W=1 NRCL=5 EHR_ENV=Mayo RUN_TAG=bal1 python aggregate_results.py
    MODEL=CRUX EHR_ENV=CCHS USE_HALIDE_W=1 DATA_MODE=all RUN_TAG=v1 python aggregate_results.py
    MODEL=CRUX EHR_ENV=CCHS USE_HALIDE_W=1 GP_CREDIT=0 DATA_MODE=all RUN_TAG=nogp python aggregate_results.py
    MODEL=CRUX EHR_ENV=CCHS USE_HALIDE_W=1 USE_CRITICALITY=0 DATA_MODE=all RUN_TAG=nocrit python aggregate_results.py
"""
import os
import sys
import csv
import numpy as np

env            = os.getenv('EHR_ENV', 'CCHS').strip()
use_halide_w   = os.getenv('USE_HALIDE_W', '0').strip() in ('1', 'true', 'yes')
model          = os.getenv('MODEL', 'THEMES').strip()   # THEMES | EM_EDM | EDM | IQLearn | PWDICE | CHALIDE | SOIL | THEMES_CD | CRUX | CCA
use_suboptimal = os.getenv('USE_SUBOPTIMAL', '0').strip() in ('1', 'true', 'yes')
cleantrain     = os.getenv('CLEANTRAIN', '0').strip() in ('1', 'true', 'yes')
# IQLearn, PWDICE, CHALIDE, SOIL, THEMES_CD, CRUX, and CCA always write cleantrain-style filenames
if model in ('IQLearn', 'PWDICE', 'CHALIDE', 'SOIL', 'THEMES_CD', 'CRUX', 'CCA'):
    cleantrain = True
_data_mode_env  = os.getenv('DATA_MODE', '').strip()
data_tag        = _data_mode_env if _data_mode_env in ('e', 'ei', 'en', 'all') else ('ei' if use_suboptimal else 'e')
_weight_key     = os.getenv('WEIGHT_KEY', 'halide_weights').strip()
weight_tag      = ('unweighted' if not use_halide_w else
                   ('weighted' if _weight_key == 'halide_weights' else _weight_key))
_halide_variant = os.getenv('HALIDE_VARIANT', '').strip()
variant_tag     = f'_{_halide_variant}' if _halide_variant else ''
_nrcl           = int(os.getenv('NRCL', '3'))
nrcl_tag        = f'_nrcl{_nrcl}' if _nrcl != 3 else ''
repeat_num      = int(os.getenv('REPEAT_NUM', '10'))

# CHALIDE filenames include a dice tag: results_halide_cleantrain_{data}_{weight}_{dice}_rp{N}.csv
# SOIL and THEMES_CD filenames do NOT include a dice tag: results_halide_cleantrain_{data}_{weight}_rp{N}.csv
_use_dice  = os.getenv('USE_DICE', '1').strip() in ('1', 'true', 'yes')
dice_tag   = ('dice' if _use_dice else 'nodice') if model == 'CHALIDE' else ''

# RUN_TAG: matches the tag used when training — empty = original filenames
_run_tag  = os.getenv('RUN_TAG', '').strip()
tag_suffix = f'_{_run_tag}' if _run_tag else ''

# CRUX/CCA-specific tags (only used when MODEL=CRUX or MODEL=CCA)
_use_gp_credit     = os.getenv('GP_CREDIT',       '1').strip() in ('1', 'true', 'yes')
_use_criticality   = os.getenv('USE_CRITICALITY',  '1').strip() in ('1', 'true', 'yes')
_lambda_contrast   = float(os.getenv('LAMBDA_CONTRAST', '0.5'))

base_path      = os.path.dirname(os.path.abspath(__file__)) + '/'
results_dir    = base_path + f'results/EHRs/{env}/{model}/'

# File name pattern matches what each training script writes.
# Standard layout: {prefix}_{data}_{weight}{variant}{nrcl}{dice_suffix}{run_tag}_rp{N}.csv
# CRUX layout:     results_crux_cleantrain_{data}_{weight}{crit_tag}{contrast_tag}{variant}{nrcl}{run_tag}_rp{N}.csv
# CCA  layout:     results_cca_cleantrain_{data}_{weight}{crit_tag}{contrast_tag}{variant}{nrcl}{run_tag}_rp{N}.csv
def _make_patterns(cleantrain, data_tag, weight_tag, variant_tag, nrcl_tag, dice_tag, tag_suffix=''):
    dice_suffix = f'_{dice_tag}' if dice_tag else ''
    prefix = 'results_halide_cleantrain' if cleantrain else 'results_halide'
    body = f'{prefix}_{data_tag}_{weight_tag}{variant_tag}{nrcl_tag}{dice_suffix}{tag_suffix}'
    per_rep = f'{body}_rp{{rp_idx}}.csv'
    agg_out = f'{body}_aggregated.csv'
    return per_rep, agg_out

def _make_crux_patterns(data_tag, weight_tag, variant_tag, nrcl_tag, tag_suffix,
                        use_criticality, lambda_contrast, prefix='crux'):
    crit_tag     = '_crit' if use_criticality else ''
    contrast_tag = f'_lc{lambda_contrast}' if lambda_contrast > 0 else ''
    body = (f'results_{prefix}_cleantrain_{data_tag}_{weight_tag}'
            f'{crit_tag}{contrast_tag}{variant_tag}{nrcl_tag}{tag_suffix}')
    per_rep = f'{body}_rp{{rp_idx}}.csv'
    agg_out = f'{body}_aggregated.csv'
    return per_rep, agg_out

if model in ('CRUX', 'CCA'):
    # weight_tag: gp_weighted when GP_CREDIT=1, else matches THEMES logic
    if not use_halide_w:
        crux_weight_tag = 'unweighted'
    elif _use_gp_credit:
        crux_weight_tag = 'gp_weighted'
    else:
        crux_weight_tag = 'weighted'
    per_rep_pattern, agg_out_name = _make_crux_patterns(
        data_tag, crux_weight_tag, variant_tag, nrcl_tag, tag_suffix,
        _use_criticality, _lambda_contrast,
        prefix=model.lower(),
    )
else:
    per_rep_pattern, agg_out_name = _make_patterns(
        cleantrain, data_tag, weight_tag, variant_tag, nrcl_tag, dice_tag, tag_suffix)

_display_weight = (crux_weight_tag if model in ('CRUX', 'CCA') else weight_tag)  # noqa: F821
print(f"[aggregate] model={model}  env={env}  data_tag={data_tag}  weight_tag={_display_weight}"
      + (f"  variant={_halide_variant}" if _halide_variant else "")
      + (f"  nrcl={_nrcl}" if _nrcl != 3 else "")
      + (f"  dice_tag={dice_tag}" if dice_tag else "")
      + (f"  gp_credit={_use_gp_credit}  criticality={_use_criticality}  lambda_contrast={_lambda_contrast}" if model in ('CRUX', 'CCA') else "")
      + (f"  run_tag={_run_tag}" if _run_tag else "")
      + f"  cleantrain={cleantrain}")
print(f"[aggregate] results_dir={results_dir}")
print(f"[aggregate] per_rep_pattern={per_rep_pattern}")

header = ('Round', 'Accuracy', 'Recall', 'Precision', 'F-score', 'AUC', 'APR', 'Jaccard')
repeat_results = []
missing        = []

for rp_idx in range(repeat_num):
    fpath = os.path.join(results_dir, per_rep_pattern.format(rp_idx=rp_idx))
    if not os.path.exists(fpath):
        missing.append(rp_idx)
        continue
    with open(fpath, newline='') as f:
        rows = list(csv.reader(f))
    # THEMES files contain one row per outer iteration (rp{N}_iter{k}) plus a
    # rp{N}_best summary row.  EDM/EM-EDM files have a single data row per repeat.
    # Prefer the explicit _best row; fall back to the last non-header/non-final row
    # for backward compatibility with single-row format files.
    data_rows = [r for r in rows if r and r[0] not in ('Round', 'final')]
    _iter_sel = os.getenv('ITER_SEL', 'best').strip().lower()  # 'best' (oracle) | 'last'
    best_row = next((r for r in data_rows if str(r[0]).endswith('_best')), None) if _iter_sel == 'best' else None
    chosen_row = best_row if best_row is not None else (data_rows[0] if data_rows else None)
    if chosen_row is not None:
        repeat_results.append([float(v) for v in chosen_row[1:]])  # skip Round col
        print(f"  rp{rp_idx}: {chosen_row[1:]}  (from row '{chosen_row[0]}')")
    else:
        missing.append(rp_idx)

if missing:
    print(f"\nWARNING: missing results for repeats: {missing}")

if not repeat_results:
    print("No results found. Check that job array tasks completed.")
    sys.exit(1)

arr        = np.array(repeat_results)
avg_result = np.mean(arr, axis=0)
std_result = np.std(arr, axis=0)
agg_result = [f'{avg_result[i]:.3f}({std_result[i]:.2f})' for i in range(len(avg_result))]

out_path = os.path.join(results_dir, agg_out_name)
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for rp_idx, row in enumerate(repeat_results):
        writer.writerow([rp_idx] + [f'{v:.4f}' for v in row])
    writer.writerow(['final'] + agg_result)

print(f"\nAggregated {len(repeat_results)}/{repeat_num} repeats → {out_path}")
print(f"Final: {dict(zip(header[1:], agg_result))}")
