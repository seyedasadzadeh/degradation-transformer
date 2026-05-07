import numpy as np

def forecast_metrics(
    learner,
    data,
    num_periods=60,
    context_window=None,
    batch_size=256,
    stride=1,
    max_samples=None,
    seed=42,
    temperature=0.0,
    return_predictions=False,
):
    """
    Evaluate autoregressive forecasts over sliding windows from full episodes.

    For each valid start position, the function uses
    data[:, start:start+context_window] as context and compares the rollout to
    data[:, start+context_window:start+context_window+num_periods].
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array with shape (n_episodes, episode_length).")

    context_window = int(context_window or learner.model.context_window)
    num_periods = int(num_periods)
    if context_window <= 0 or num_periods <= 0:
        raise ValueError("context_window and num_periods must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    n_episodes, episode_length = data.shape
    max_start = episode_length - context_window - num_periods
    if max_start < 0:
        raise ValueError(
            "Episodes are too short for the requested context and forecast horizon: "
            f"episode_length={episode_length}, context_window={context_window}, num_periods={num_periods}."
        )

    starts = np.arange(0, max_start + 1, stride)
    episode_idx, start_idx = np.meshgrid(np.arange(n_episodes), starts, indexing="ij")
    sample_index = np.column_stack([episode_idx.ravel(), start_idx.ravel()])

    if max_samples is not None and len(sample_index) > max_samples:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(sample_index), size=int(max_samples), replace=False)
        sample_index = sample_index[chosen]

    n_samples = len(sample_index)
    per_horizon_sse = np.zeros(num_periods, dtype=np.float64)
    per_horizon_sae = np.zeros(num_periods, dtype=np.float64)
    total_error = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    final_abs_error = 0.0
    final_sq_error = 0.0
    slope_abs_error = 0.0
    slope_sq_error = 0.0
    slope_count = 0

    contexts_out = []
    y_true_out = []
    y_pred_out = []
    indices_out = []

    for batch_start in range(0, n_samples, batch_size):
        batch_index = sample_index[batch_start:batch_start + batch_size]
        contexts = np.stack([
            data[e, s:s + context_window]
            for e, s in batch_index
        ]).astype(np.float32)
        y_true = np.stack([
            data[e, s + context_window:s + context_window + num_periods]
            for e, s in batch_index
        ]).astype(np.float32)

        y_pred = learner.predict(contexts, num_periods=num_periods, temperature=temperature)[0]
        y_pred = np.asarray(y_pred, dtype=np.float32)

        error = y_pred - y_true
        abs_error = np.abs(error)
        sq_error = error ** 2

        per_horizon_sse += sq_error.sum(axis=0)
        per_horizon_sae += abs_error.sum(axis=0)
        total_error += error.sum()
        total_abs_error += abs_error.sum()
        total_sq_error += sq_error.sum()
        final_abs_error += abs_error[:, -1].sum()
        final_sq_error += sq_error[:, -1].sum()

        true_with_anchor = np.concatenate([contexts[:, -1:], y_true], axis=1)
        pred_with_anchor = np.concatenate([contexts[:, -1:], y_pred], axis=1)
        slope_error = np.diff(pred_with_anchor, axis=1) - np.diff(true_with_anchor, axis=1)
        slope_abs_error += np.abs(slope_error).sum()
        slope_sq_error += (slope_error ** 2).sum()
        slope_count += slope_error.size

        if return_predictions:
            contexts_out.append(contexts)
            y_true_out.append(y_true)
            y_pred_out.append(y_pred)
            indices_out.append(batch_index)

    value_count = n_samples * num_periods
    metrics = {
        "n_samples": int(n_samples),
        "n_episodes": int(n_episodes),
        "episode_length": int(episode_length),
        "context_window": int(context_window),
        "num_periods": int(num_periods),
        "stride": int(stride),
        "mse": float(total_sq_error / value_count),
        "rmse": float(np.sqrt(total_sq_error / value_count)),
        "mae": float(total_abs_error / value_count),
        "bias": float(total_error / value_count),
        "final_mae": float(final_abs_error / n_samples),
        "final_rmse": float(np.sqrt(final_sq_error / n_samples)),
        "slope_mae": float(slope_abs_error / slope_count),
        "slope_rmse": float(np.sqrt(slope_sq_error / slope_count)),
        "per_horizon_mse": per_horizon_sse / n_samples,
        "per_horizon_mae": per_horizon_sae / n_samples,
    }

    if return_predictions:
        metrics["sample_index"] = np.concatenate(indices_out, axis=0)
        metrics["x_context"] = np.concatenate(contexts_out, axis=0)
        metrics["y_true"] = np.concatenate(y_true_out, axis=0)
        metrics["y_pred"] = np.concatenate(y_pred_out, axis=0)

    return metrics

# ----------------------------------------------------------------------------------------------
# 
#    
#------------------------------------------- RLHF Components -----------------------------------
#
#
# ----------------------------------------------------------------------------------------------
