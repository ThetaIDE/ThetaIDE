"""
Contrastive Critical State Scoring
====================================
Computes a per-MT-TICC-cluster criticality score C(k) in [0, 1] by measuring
the Jensen-Shannon Divergence (JSD) between the action distributions of
expert-positive and anti-expert-negative demonstrations at each cluster.

Core idea
---------
  C(k) = JSD( P_+(a | cluster=k) || P_-(a | cluster=k) ) / log(2)

where:
  P_+(a | k)  = empirical action distribution over timesteps from
                expert-positive trajectories (shock->non-shock, Q10)
                that are assigned to cluster k by MT-TICC.
  P_-(a | k)  = empirical action distribution over timesteps from
                anti-expert-negative trajectories (shock->shock, Q11)
                that are assigned to cluster k.

JSD is normalised by log(2) so C(k) in [0, 1].
  C(k) = 1  -> experts and anti-experts choose completely different actions at k
  C(k) = 0  -> identical action distributions (non-critical)

Clusters with fewer than min_support samples from either group fall back to
the mean criticality of well-supported clusters.

Key functions
-------------
compute_contrastive_criticality(norm_agg_split_df, idx_list, ticc_labels,
                                 TICC_clusters, action_col, min_support)
    Main entry point.  Returns a (TICC_clusters,) array.

criticality_from_counts(pos_counts, neg_counts, min_support)
    Lower-level helper; operates on pre-accumulated count matrices.

Usage
-----
    from contrastive_criticality import compute_contrastive_criticality

    # ticc_labels: list of arrays, one per trajectory in norm_agg_split_df
    # idx_list:    [Q00_idx, Q10_idx, Q01_idx, Q11_idx]
    criticality = compute_contrastive_criticality(
        norm_agg_split_df, idx_list, ticc_labels, TICC_clusters=9
    )
    # criticality[k] in [0, 1] for each TICC cluster k
"""

import numpy as np
from scipy.special import rel_entr
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# JSD utilities
# ---------------------------------------------------------------------------

def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Jensen-Shannon Divergence (base e) between two probability vectors p and q.
    Both are smoothed by eps to avoid log(0).

    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M),  M = 0.5*(P+Q)

    Returns a non-negative float; maximum is log(2) for completely disjoint supports.
    """
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m)))


def _normalised_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """JSD normalised to [0, 1] by dividing by log(2)."""
    return _jsd(p, q) / np.log(2.0)


# ---------------------------------------------------------------------------
# Count accumulation
# ---------------------------------------------------------------------------

def _accumulate_action_counts(
    norm_agg_split_df: list,
    ticc_labels: list,
    traj_indices: set,
    n_clusters: int,
    n_actions: int,
    action_col: str = 'Action',
) -> np.ndarray:
    """
    Accumulate action counts per TICC cluster over a subset of trajectories.

    Parameters
    ----------
    norm_agg_split_df : list of DataFrames
        One DataFrame per trajectory (globally indexed).
    ticc_labels : list of array-like
        TICC cluster assignment per timestep, globally indexed to match
        norm_agg_split_df.
    traj_indices : set of int
        Global trajectory indices to include (e.g. Q10 or Q11 indices).
    n_clusters : int
        Number of TICC clusters K.
    n_actions : int
        Number of discrete actions A.
    action_col : str
        Column in each trajectory DataFrame that holds the discrete action.

    Returns
    -------
    counts : np.ndarray, shape (n_clusters, n_actions)
        counts[k, a] = number of timesteps in cluster k with action a,
        over the selected trajectories.
    """
    counts = np.zeros((n_clusters, n_actions), dtype=float)
    for i, (df, labels) in enumerate(zip(norm_agg_split_df, ticc_labels)):
        if i not in traj_indices:
            continue
        actions = np.array(df[action_col], dtype=int)
        labels_arr = np.array(labels, dtype=int)
        T = min(len(actions), len(labels_arr))
        for t in range(T):
            k = int(labels_arr[t])
            a = int(np.clip(actions[t], 0, n_actions - 1))
            if 0 <= k < n_clusters:
                counts[k, a] += 1.0
    return counts


# ---------------------------------------------------------------------------
# Criticality from pre-accumulated counts
# ---------------------------------------------------------------------------

def criticality_from_counts(
    pos_counts: np.ndarray,
    neg_counts: np.ndarray,
    min_support: int = 5,
) -> np.ndarray:
    """
    Compute per-cluster criticality from pre-accumulated count matrices.

    Parameters
    ----------
    pos_counts : np.ndarray, shape (K, A)
        Action counts for expert-positive trajectories per cluster.
    neg_counts : np.ndarray, shape (K, A)
        Action counts for anti-expert-negative trajectories per cluster.
    min_support : int
        Minimum number of timestep samples from EACH group required for a
        cluster to receive a data-derived criticality score.  Clusters below
        this threshold receive the fallback mean criticality.

    Returns
    -------
    criticality : np.ndarray, shape (K,)
        Normalised JSD in [0, 1] per cluster.
    """
    n_clusters = pos_counts.shape[0]
    raw = np.full(n_clusters, -1.0)   # -1 = sentinel for low-support fallback

    for k in range(n_clusters):
        n_pos = pos_counts[k].sum()
        n_neg = neg_counts[k].sum()
        if n_pos < min_support or n_neg < min_support:
            continue
        p_pos = pos_counts[k] / n_pos
        p_neg = neg_counts[k] / n_neg
        raw[k] = _normalised_jsd(p_pos, p_neg)

    # Fallback: use mean over well-supported clusters; if none exist, use 0.5
    valid = raw[raw >= 0.0]
    fallback = float(valid.mean()) if len(valid) > 0 else 0.5
    criticality = np.where(raw >= 0.0, raw, fallback)
    # Normalize so max(C)=1, making relative differences meaningful
    # regardless of raw JSD magnitude (which is typically very small).
    max_c = criticality.max()
    if max_c > 1e-8:
        criticality = criticality / max_c
    return criticality


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_contrastive_criticality(
    norm_agg_split_df: list,
    idx_list: list,
    ticc_labels: list,
    TICC_clusters: int = 9,
    action_col: str = 'Action',
    min_support: int = 5,
    n_actions: int = 2,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute contrastive criticality scores C(k) for all MT-TICC clusters.

    Parameters
    ----------
    norm_agg_split_df : list of DataFrames, length N
        Normalised trajectory DataFrames (globally indexed 0..N-1).
    idx_list : list of 4 lists
        [Q00_indices, Q10_indices, Q01_indices, Q11_indices]
        Q10 = shock->non-shock (expert positive)
        Q11 = shock->shock     (anti-expert negative)
    ticc_labels : list of array-like, length N
        TICC cluster label sequence per trajectory (global indexing).
        When using inductive TICC (per-repeat), pass the training-fold labels;
        call this function again after re-running TICC each iteration if labels
        change.
    TICC_clusters : int
        Number of TICC clusters K.
    action_col : str
        Column name for the discrete action in each trajectory DataFrame.
    min_support : int
        Minimum per-group support per cluster for data-derived criticality.
    n_actions : int
        Cardinality of the action space.  Binary actions: n_actions=2.
    verbose : bool
        Print per-cluster criticality summary.

    Returns
    -------
    criticality : np.ndarray, shape (TICC_clusters,)
        C(k) in [0, 1] for each cluster k.
    """
    pos_idx = set(idx_list[1])   # Q10: shock -> non-shock (expert)
    neg_idx = set(idx_list[3])   # Q11: shock -> shock     (anti-expert)

    pos_counts = _accumulate_action_counts(
        norm_agg_split_df, ticc_labels, pos_idx,
        TICC_clusters, n_actions, action_col,
    )
    neg_counts = _accumulate_action_counts(
        norm_agg_split_df, ticc_labels, neg_idx,
        TICC_clusters, n_actions, action_col,
    )

    criticality = criticality_from_counts(pos_counts, neg_counts, min_support)

    if verbose:
        print("[CRUX] Contrastive criticality scores per TICC cluster:")
        for k in range(TICC_clusters):
            ps = pos_counts[k].sum()
            ns = neg_counts[k].sum()
            flag = '' if (ps >= min_support and ns >= min_support) else ' [fallback]'
            print(f"  k={k:2d}: C={criticality[k]:.4f}  "
                  f"(pos_support={int(ps)}, neg_support={int(ns)}){flag}")

    return criticality


# ---------------------------------------------------------------------------
# Per-repeat update helper
# ---------------------------------------------------------------------------

def update_criticality_inductive(
    slice_tr_agg_df: list,
    ticc_train_idx: list,
    idx_list: list,
    tr_ticc_labels: list,
    TICC_clusters: int = 9,
    action_col: str = 'Action',
    min_support: int = 5,
    n_actions: int = 2,
    verbose: bool = False,
) -> np.ndarray:
    """
    Convenience wrapper for inductive TICC: computes criticality using only
    the training-fold trajectories and their locally-indexed TICC labels.

    In inductive mode, tr_ticc_labels[j] gives the TICC label sequence for
    the j-th trajectory in slice_tr_agg_df, where j corresponds to global
    index ticc_train_idx[j].

    Parameters
    ----------
    slice_tr_agg_df : list of DataFrames
        Training-fold trajectory DataFrames (locally indexed 0..M-1).
    ticc_train_idx : list of int
        Global indices of training trajectories: global_i = ticc_train_idx[j].
    idx_list : list of 4 lists
        Global quadrant indices [Q00, Q10, Q01, Q11].
    tr_ticc_labels : list of array-like, length M
        TICC labels for the training fold (locally indexed).

    Returns
    -------
    criticality : np.ndarray, shape (TICC_clusters,)
    """
    # Build a globally-indexed mapping for the training slice
    pos_idx_global = set(idx_list[1])
    neg_idx_global = set(idx_list[3])

    # Convert to local training indices
    pos_idx_local = {j for j, gi in enumerate(ticc_train_idx) if gi in pos_idx_global}
    neg_idx_local = {j for j, gi in enumerate(ticc_train_idx) if gi in neg_idx_global}

    local_idx_list = [[], list(pos_idx_local), [], list(neg_idx_local)]

    return compute_contrastive_criticality(
        norm_agg_split_df=slice_tr_agg_df,
        idx_list=local_idx_list,
        ticc_labels=tr_ticc_labels,
        TICC_clusters=TICC_clusters,
        action_col=action_col,
        min_support=min_support,
        n_actions=n_actions,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Continuous per-timestep criticality via kNN local action distribution JSD
# ---------------------------------------------------------------------------

def build_knn_criticality_index(
    slice_tr_agg_df: list,
    ticc_train_idx: list,
    idx_list: list,
    feature_cols: list,
    action_col: str = 'Action',
    n_neighbors: int = 50,
    n_actions: int = 2,
    pos_quadrant_idx: int = 1,
    neg_quadrant_idx: int = 3,
    verbose: bool = True,
):
    """
    Build kNN indexes over the chosen positive and anti-expert quadrants.

    Each timestep is represented by its fixed-dimensional clinical feature vector.
    Variable-length trajectories are not a problem: each timestep is an independent
    d-dimensional point; trajectory length has no effect on the index.

    Parameters
    ----------
    slice_tr_agg_df : list of DataFrames, locally indexed 0..M-1
    ticc_train_idx  : global indices of the M training trajectories
    idx_list        : [Q00, Q10, Q01, Q11] global quadrant indices
    feature_cols    : feature columns used for kNN distance (e.g. 14 clinical features)
    n_neighbors     : k for kNN lookup
    n_actions       : action-space cardinality
    pos_quadrant_idx: index into idx_list for the positive/expert quadrant
                      1 = Q10 (shock→nonshock, default), 0 = Q00 (nonshock→nonshock)
    neg_quadrant_idx: index into idx_list for the anti-expert quadrant
                      2 = Q01 (nonshock→shock), 3 = Q11 (shock→shock, default)

    Returns
    -------
    nn_pos, pos_actions, nn_neg, neg_actions
        sklearn NearestNeighbors objects and corresponding action arrays.
    """
    pos_idx_global = set(idx_list[pos_quadrant_idx])  # configurable expert
    neg_idx_global = set(idx_list[neg_quadrant_idx])   # configurable anti-expert

    pos_feats_list, pos_acts_list = [], []
    neg_feats_list, neg_acts_list = [], []

    for gi, df in zip(ticc_train_idx, slice_tr_agg_df):
        feats = np.array(df[feature_cols], dtype=float)
        acts  = np.clip(np.array(df[action_col], dtype=int), 0, n_actions - 1)
        if gi in pos_idx_global:
            pos_feats_list.append(feats)
            pos_acts_list.append(acts)
        if gi in neg_idx_global:
            neg_feats_list.append(feats)
            neg_acts_list.append(acts)

    pos_features = (np.vstack(pos_feats_list) if pos_feats_list
                    else np.empty((0, len(feature_cols)), dtype=float))
    pos_actions  = (np.concatenate(pos_acts_list) if pos_acts_list
                    else np.array([], dtype=int))
    neg_features = (np.vstack(neg_feats_list) if neg_feats_list
                    else np.empty((0, len(feature_cols)), dtype=float))
    neg_actions  = (np.concatenate(neg_acts_list) if neg_acts_list
                    else np.array([], dtype=int))

    # Balance Q10 and anti-expert pools so the larger group does not dominate
    # the JSD criticality distribution (Q11 is ~2× Q10 in CCHS).
    # Subsample the larger pool to match the smaller using a fixed RNG seed
    # so criticality values are reproducible across outer-loop iterations.
    n_pos, n_neg = len(pos_features), len(neg_features)
    if n_pos > 0 and n_neg > 0 and n_pos != n_neg:
        rng_bal = np.random.default_rng(seed=0)
        target  = min(n_pos, n_neg)
        if n_neg > n_pos:
            sel          = rng_bal.choice(n_neg, size=target, replace=False)
            neg_features = neg_features[sel]
            neg_actions  = neg_actions[sel]
        else:
            sel          = rng_bal.choice(n_pos, size=target, replace=False)
            pos_features = pos_features[sel]
            pos_actions  = pos_actions[sel]

    k_pos = min(n_neighbors, max(1, len(pos_features)))
    k_neg = min(n_neighbors, max(1, len(neg_features)))

    nn_pos = (NearestNeighbors(n_neighbors=k_pos, metric='euclidean',
                                algorithm='ball_tree').fit(pos_features)
              if len(pos_features) > 0 else None)
    nn_neg = (NearestNeighbors(n_neighbors=k_neg, metric='euclidean',
                                algorithm='ball_tree').fit(neg_features)
              if len(neg_features) > 0 else None)

    _quad_label = {0: 'Q00', 1: 'Q10', 2: 'Q01', 3: 'Q11'}
    _pos_label = _quad_label.get(pos_quadrant_idx, f'Q[{pos_quadrant_idx}]')
    _neg_label = _quad_label.get(neg_quadrant_idx, f'Q[{neg_quadrant_idx}]')
    if verbose:
        print(f'[kNN] criticality index built: '
              f'{len(pos_features)} {_pos_label} timesteps (pos), '
              f'{len(neg_features)} {_neg_label} timesteps (neg), '
              f'k_pos={k_pos}, k_neg={k_neg}')

    return nn_pos, pos_actions, nn_neg, neg_actions


def compute_all_continuous_criticalities(
    slice_tr_agg_df: list,
    feature_cols: list,
    action_col: str,
    nn_pos,
    pos_actions: np.ndarray,
    nn_neg,
    neg_actions: np.ndarray,
    n_actions: int = 2,
    min_support: int = 3,
    weight_cap: float = 2.0,
    soft_cap: bool = True,
    verbose: bool = True,
):
    """
    Compute per-timestep continuous criticality for all training trajectories.

    For each timestep t with feature vector s_t:
      C(s_t)       = JSD( P+(a|s_t) || P-(a|s_t) )   [symmetric — for EM-EDM weights]
      p_pos(s_t, a_t) = P+(a_t | s_t)                [expert action probability at s_t]
      p_neg(s_t, a_t) = P-(a_t | s_t)                [anti-expert action probability at s_t]

    P+(a|s_t) is estimated from the k nearest Q10 neighbors of s_t in feature space;
    P-(a|s_t) from the k nearest Q11 neighbors.  This is the continuous-state
    generalisation of the cluster-level JSD used in compute_contrastive_criticality.

    C values are globally normalised so max(C)=1 across all training timesteps.

    Returns
    -------
    crit_ts  : list of np.ndarray — C(s_t) in [0,1] per trajectory
    p_pos_ts : list of np.ndarray — P+(a_t|s_t) for the actual action taken
    p_neg_ts : list of np.ndarray — P-(a_t|s_t) for the actual action taken
    """
    crit_raw, p_pos_ts_raw, p_neg_ts_raw = [], [], []

    for df in slice_tr_agg_df:
        feats = np.array(df[feature_cols], dtype=float)
        acts  = np.clip(np.array(df[action_col], dtype=int), 0, n_actions - 1)
        T = len(feats)

        c_arr  = np.full(T, -1.0)
        pp_arr = np.full(T, 1.0 / n_actions)
        pn_arr = np.full(T, 1.0 / n_actions)

        if T == 0 or nn_pos is None or nn_neg is None:
            crit_raw.append(c_arr)
            p_pos_ts_raw.append(pp_arr)
            p_neg_ts_raw.append(pn_arr)
            continue

        _, idx_pos = nn_pos.kneighbors(feats)   # (T, k_pos)
        _, idx_neg = nn_neg.kneighbors(feats)   # (T, k_neg)

        for t in range(T):
            pc = np.bincount(pos_actions[idx_pos[t]], minlength=n_actions).astype(float)
            nc = np.bincount(neg_actions[idx_neg[t]], minlength=n_actions).astype(float)
            n_p, n_n = pc.sum(), nc.sum()

            if n_p >= min_support and n_n >= min_support:
                c_arr[t] = _normalised_jsd(pc / n_p, nc / n_n)

            # Laplace-smoothed action probabilities for the actual action taken
            a_t = int(acts[t])
            eps = 1e-6
            pp_arr[t] = ((pc[a_t] + eps) / (n_p + n_actions * eps)
                         if n_p > 0 else 1.0 / n_actions)
            pn_arr[t] = ((nc[a_t] + eps) / (n_n + n_actions * eps)
                         if n_n > 0 else 1.0 / n_actions)

        crit_raw.append(c_arr)
        p_pos_ts_raw.append(pp_arr)
        p_neg_ts_raw.append(pn_arr)

    # Global normalisation: fill sentinel −1 with mean, scale so mean(C)=1.
    # Normalising to mean=1 (rather than max=1) keeps the average EM-EDM weight
    # unchanged from the GP-only baseline while still boosting critical states.
    # max=1 normalisation produces mean≈0.046, which crushes the training signal.
    all_valid   = [v for c in crit_raw for v in c if v >= 0.0]
    global_mean = float(np.mean(all_valid)) if all_valid else 1.0
    fallback    = global_mean  # unsupported timesteps get the mean weight (×1 after scaling)
    scale       = 1.0 / global_mean if global_mean > 1e-8 else 1.0

    # Compress weights toward weight_cap.
    #
    # soft_cap=True  (default): tanh compression — weight_cap * tanh(x / weight_cap)
    #   Maps [0, ∞) → [0, weight_cap) smoothly. For x << cap the output is ≈ x
    #   (linear, ordering preserved). For x >> cap it saturates asymptotically.
    #   No two timesteps collapse to the same value, so the continuous signal survives.
    #
    # soft_cap=False: hard clip to [0, weight_cap]. Faster but creates a spike at cap
    #   where all values above the threshold are treated identically — bimodal on small
    #   datasets where many timesteps exceed the ceiling.

    crit_ts = []
    for c_arr in crit_raw:
        filled = np.where(c_arr >= 0.0, c_arr, fallback) * scale
        if soft_cap:
            compressed = weight_cap * np.tanh(filled / weight_cap)
        else:
            compressed = np.clip(filled, 0.0, weight_cap)
        crit_ts.append(compressed)

    if verbose:
        all_c = np.concatenate(crit_ts)
        mode_str = 'soft(tanh)' if soft_cap else 'hard'
        print(f'[CRUX] Continuous criticality computed over '
              f'{sum(len(c) for c in crit_ts)} timesteps: '
              f'mean={all_c.mean():.3f}, max={all_c.max():.3f}, '
              f'min={all_c.min():.3f}, cap={weight_cap:.3f}({mode_str})')

    return crit_ts, p_pos_ts_raw, p_neg_ts_raw
