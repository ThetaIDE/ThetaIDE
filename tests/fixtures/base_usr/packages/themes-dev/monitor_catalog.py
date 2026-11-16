"""Headless declarations for numerical AL diagnostics; standard library only."""


def outputs(config, context):
    rows = []
    def scalar(name, label, group):
        rows.append({'name': name, 'label': label, 'group': group})
    for name, label in [('iteration', 'TICC iteration'), ('changed_fraction', 'Stage assignments changed'),
                        ('active_stages', 'Active stages'), ('solver_primal_residual', 'Precision solver primal residual'),
                        ('solver_dual_residual', 'Precision solver dual residual'), ('solver_converged_fraction', 'Precision solvers converged')]:
        scalar('ticc/' + name, label, 'TICC')
    model=context.get('model') or {}
    stage=model.get('stage',model)
    stage=stage.get('params',stage)
    for stage in range(int(stage.get('stages', 5))):
        scalar(f'ticc/stage_{stage}/support', f'Current stage {stage} support', 'TICC')
    for name, label in [('round', 'EM round'), ('nll', 'Training mixture NLL per scored row'),
                        ('changed_fraction', 'Policy assignments changed'), ('active_policies', 'Active policies')]:
        scalar('em/' + name, label, 'EM')
    for policy in range(int((config.get('em') or {}).get('groups',config.get('groups',3)))):
        for name in ('support', 'prior'):
            scalar(f'em/policy_{policy}/{name}', f'Policy {policy} {name}', 'EM')
        for name in ('loss', 'cross_entropy', 'density', 'sgld_gradient_norm', 'negative_energy', 'update', 'round'):
            scalar(f'edm/policy_{policy}/{name}', f'Policy {policy} {name.replace("_", " ")}', 'EDM')
    for name in ('outer_iteration','stage_partition_disagreement','segments','active_policies','train_nll','heldout_nll','irl_initial_nll','irl_final_nll','reward_range','feedback_std','optimizer_updates'):
        scalar('themes/'+name,name.replace('_',' '),'THEMES outer loop')
    rows.extend(visual_outputs())
    return rows


def visual_outputs():
    return [dict(row, visual_schema='theta.visual-output/v1', artifact_schema='themes.diagnostics/v1',
                          retention='snapshots', requires=['numerical_round.v1'], group='THEMES diagnostics') for row in [
        {'name': 'themes/policy_response', 'label': 'Policy action probabilities on fixed training probes',
         'kind': 'matrix', 'axes': ['probe', 'policy'], 'value_semantics': 'Binary action probability on frozen training inputs'},
        {'name': 'themes/feature_ablation', 'label': 'Policy response to feature replacement',
         'kind': 'matrix', 'axes': ['feature', 'policy'], 'value_semantics': 'Mean probability change after training-median replacement; not a causal treatment effect'},
        {'name': 'themes/training_membership', 'label': 'Training segment membership confidence',
         'kind': 'distribution', 'axes': ['confidence'], 'value_semantics': 'Maximum retrospective training-segment membership probability'}]]
