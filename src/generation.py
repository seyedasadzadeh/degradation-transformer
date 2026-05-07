import numpy as np

class BaseDegradationProcess:
    def __init__(self, length, dim):
        self.length = int(length)
        self.dim = int(dim)

    def generate_episode(self, x0):
        x0 = np.atleast_1d(np.asarray(x0))
        episode = np.zeros((x0.shape[0], self.length))
        episode[:, 0] = x0
        for i in range(self.length-1):
            episode[:, i + 1] = episode[:, i] + self.xdot(episode[:, i])
        return episode


class ParisLawDegradation(BaseDegradationProcess):
    """
    Paris–Erdogan fatigue crack growth model.
    Paris’ law (fatigue crack growth)
    $dX = C\,(\pi X)^{m/2}\,\sigma^m\,dt$  (or simplified $dX = k X^{m/2}\,dt$)

    """

    def __init__(self, length, dim, c=1e-12, m=3):
        super().__init__(length, dim)
        self.c = float(c)
        self.m = float(m)

    def xdot(self, a):
        a = np.atleast_1d(np.asarray(a))
        return self.c * a ** (self.m/2)

class SEILayer(BaseDegradationProcess):
    """
    A general class for these degradation mechanisms:
     - SEI layer growth (diffusion-limited)
    $dX = \dfrac{k}{X}\,dt$

    - Dead lithium formation
    $dX = \gamma_0\,\dfrac{X_0}{X}\,c_{\text{Li}}\,dt$


    """

    def __init__(self, length, dim, k, sigma_e=0):
        super().__init__(length, dim)
        self.k = float(k)
        self.sigma_e = float(sigma_e)
        self.noise = (sigma_e > 0)  # Auto-detect if noise
        
    def xdot(self, a):
        a = np.atleast_1d(np.asarray(a))
        e = np.random.randn() * self.sigma_e if self.noise else 0
        return self.k / a * (1 + e) # multiplicative noise


class LogisticStiffness(BaseDegradationProcess):
    """
    A general class for these degradation mechanisms:
     Stiffness degradation (logistic)
     $dX = -\alpha X\,(1-X/X_{\max})\,dt$


    """

    def __init__(self, length, dim, alfa, xmax, sigma_e=0):
        super().__init__(length, dim)
        self.alfa = float(alfa)
        self.xmax = float(xmax)
        self.sigma_e = float(sigma_e)
        self.noise = (sigma_e > 0)  # Auto-detect if noise
        
    def xdot(self, a):
        a = np.atleast_1d(np.asarray(a))
        fx = self.alfa * a * (1 - a / self.xmax)
        e = np.random.randn() * self.sigma_e if self.noise else 0
        return fx + e
    
class LogLogisticStiffness(BaseDegradationProcess):
    """
    A general class for these degradation mechanisms:
     Log-logistic seal degradation
     $dX = beta (x/alpha)^{1-k} / (1 + (x/alpha)^{k-1}) dt$


    """

    def __init__(self, length, dim, alfa, beta, k, sigma_e=0):
        super().__init__(length, dim)
        self.alfa = float(alfa)
        self.beta = float(beta)
        self.k = float(k)
        self.sigma_e = float(sigma_e)
        self.noise = (sigma_e > 0)  # Auto-detect if noise
        
    def xdot(self, a):
        a = np.atleast_1d(np.asarray(a))
        e = np.random.randn() * self.sigma_e if self.noise else 0
        fx = self.beta * (a/self.alfa)**(1-self.k) * (1 - a/self.alfa)** (self.k -1)
        return fx + e
    
class RandomShockDegradation(BaseDegradationProcess):
    """Gaussian time between shocks (e.g., mean=10 steps, std=3?)
        Gaussian shock magnitude
        adds also baseline (Paris or linear)
    """

    def __init__(self, length, dim, mu_t, sigma_t, mu_shock, sigma_shock, baseline=None):
        super().__init__(length, dim)
        self.mu_t = float(mu_t)
        self.sigma_t = float(sigma_t)
        self.mu_shock = float(mu_shock)
        self.sigma_shock = float(sigma_shock)
        self.baseline = baseline
        
        self.tao = int(np.random.randn()*self.sigma_t+self.mu_t)



    def xdot(self, a):
        a = np.atleast_1d(np.asarray(a))
        # assume x(t+1) = xt + bt + st
        
        self.tao = self.tao - 1 if self.tao>0 else int(np.random.randn()*self.sigma_t+self.mu_t)
        st = 0 if self.tao>0 else np.random.randn()*self.sigma_shock+self.mu_shock
        return st+self.baseline if self.baseline else st

class LinearDegradation(BaseDegradationProcess):
    """
    Linear degradation with noisy increaments
    xdot = c + e
    e ~ N(mu_e, sigma_e)
    """

    def __init__(self, length, dim, c, mu_e=0, sigma_e=0):
        super().__init__(length, dim)
        self.c = float(c)
        self.mu_e = float(mu_e)
        self.sigma_e = float(sigma_e)
        self.noise = (sigma_e > 0)  # Auto-detect if noise should be used

    def xdot(self, a):
        a = np.atleast_1d(np.asarray(a))
        e = np.random.randn() * self.sigma_e + self.mu_e if self.noise else 0
        return self.c + e


def drop_invalid(episodes):
    #drop invalid values
    episodes = episodes[~np.isnan(episodes).any(axis=1)]
    episodes = episodes[(episodes<15).all(axis=1)]
    episodes = episodes[(episodes>=0).all(axis=1)]
    return episodes

def generate_episodes_from_all_models(episode_length=100, episodes_per_param=20):

    paris_episodes = np.empty((0, episode_length))
    for c in np.arange(0.01, .1, .02):
        for m in np.arange(0.01, 4, .2):
            paris = ParisLawDegradation(length=episode_length, dim=1, c=c, m=m)
            episodes_i = paris.generate_episode(x0=np.abs(np.random.randn(episodes_per_param)*0.3+0.7))  # Initial crack lengths in meters
            paris_episodes = np.concatenate([paris_episodes, episodes_i], axis=0)
    paris_episodes = drop_invalid(paris_episodes)


    lin_episodes = np.empty((0, episode_length))
    for c in np.linspace(0.01, 0.1, 50):
        
        lin = LinearDegradation(length=episode_length, dim=1, c=c, mu_e=0, sigma_e=c/2)
        episodes_i = lin.generate_episode(x0=np.abs(np.random.randn(episodes_per_param)*0.3+0.7))  # Initial crack lengths in meters
        lin_episodes = np.concatenate([lin_episodes, episodes_i], axis=0)

    #drop invalid values
    lin_episodes = drop_invalid(lin_episodes)

    shock_episodes = np.empty((0, episode_length))
    for mu_t in range(2, 7):
        for mu_shock in np.linspace(0.1, 0.3, 5):
            shock = RandomShockDegradation(length=episode_length, dim=1, 
                                        mu_t=mu_t, sigma_t=mu_t/3, 
                                        mu_shock=mu_shock, sigma_shock=mu_shock/3, 
                                        baseline=mu_shock/10)
            
            episodes_i = shock.generate_episode(x0=np.abs(np.random.randn(episodes_per_param)*0.3+0.7))  # Initial crack lengths in meters
            shock_episodes = np.concatenate([shock_episodes, episodes_i], axis=0)
    #drop invalid values
    shock_episodes = drop_invalid(shock_episodes)

    sei_episodes = np.empty((0, episode_length))
    for k in np.linspace(0.01, 2, 100):
        
        sei = SEILayer(length=episode_length, dim=1, k=k, sigma_e=1)
        
        episodes_i = sei.generate_episode(x0=np.abs(np.random.randn(episodes_per_param)*0.3+0.7))  # Initial crack lengths in meters
        sei_episodes = np.concatenate([sei_episodes, episodes_i], axis=0)
    #drop invalid values
    sei_episodes = drop_invalid(sei_episodes)

    logistic_episodes = np.empty((0, episode_length))
    for alfa in np.linspace(0.01, .4, 40):
        
        logstiff = LogisticStiffness(length=episode_length, dim=1, alfa=alfa, xmax=15, sigma_e=alfa)
        
        episodes_i = logstiff.generate_episode(x0=np.abs(np.random.randn(episodes_per_param)*0.3+0.7))  # Initial crack lengths in meters
        logistic_episodes = np.concatenate([logistic_episodes, episodes_i], axis=0)
    #drop invalid values
    logistic_episodes = drop_invalid(logistic_episodes)

    loglogistic_episodes = np.empty((0, episode_length))
    for beta in np.linspace(0.01, .3, 8):
        for k in np.linspace(-0.9, .9, 8):
        
            logstiff = LogLogisticStiffness(length=episode_length, dim=1, alfa=15, beta=beta, k=k, sigma_e=beta)
            
            episodes_i = logstiff.generate_episode(x0=np.abs(np.random.randn(episodes_per_param)*0.3+0.7))  # Initial crack lengths in meters
            loglogistic_episodes = np.concatenate([loglogistic_episodes, episodes_i], axis=0)
    #drop invalid values
    loglogistic_episodes = drop_invalid(loglogistic_episodes)

    episodes = np.concatenate([lin_episodes, shock_episodes, paris_episodes, sei_episodes, logistic_episodes, loglogistic_episodes], axis=0)
    return episodes


# ----------------------------------------------------------------------------------------------
#
#
#------------------------------ diverse degradation episode grammar -----------------------------
#
#
# ----------------------------------------------------------------------------------------------

def _normalize_shape(shape):
    shape = np.asarray(shape, dtype=np.float64)
    shape = shape - np.nanmin(shape)
    span = np.nanmax(shape) - np.nanmin(shape)
    if not np.isfinite(span) or span < 1e-12:
        return None
    return shape / span


def _random_smooth_signal(rng, length, anchors=None):
    anchors = anchors or int(rng.integers(4, 10))
    xp = np.linspace(0, length - 1, anchors)
    yp = rng.normal(0, 1, anchors)
    signal = np.interp(np.arange(length), xp, yp)
    signal = signal - signal.mean()
    std = signal.std()
    return signal / (std + 1e-8)


def _autocorrelated_noise(rng, length, phi=None):
    phi = float(rng.uniform(0.45, 0.95) if phi is None else phi)
    eps = rng.normal(0, 1, length)
    noise = np.zeros(length)
    for i in range(1, length):
        noise[i] = phi * noise[i - 1] + eps[i]
    noise = noise - noise.mean()
    return noise / (noise.std() + 1e-8)


def _sample_changepoints(rng, min_points=1, max_points=3):
    n_points = int(rng.integers(min_points, max_points + 1))
    points = np.sort(rng.uniform(0.15, 0.85, n_points))
    return np.concatenate([[0.0], points, [1.0]])


def _piecewise_rate_shape(rng, t):
    cuts = _sample_changepoints(rng, min_points=1, max_points=4)
    rate = np.zeros_like(t)
    current = float(rng.uniform(0.01, 1.0))
    for start, end in zip(cuts[:-1], cuts[1:]):
        mask = (t >= start) & (t <= end)
        current *= float(rng.lognormal(mean=0.0, sigma=0.75))
        rate[mask] = current

    # A local knee makes the normalized shape distinct even when the total scale is removed.
    knee = float(rng.uniform(0.25, 0.8))
    knee_strength = float(rng.uniform(0.0, 4.0))
    rate = rate * (1.0 + knee_strength / (1.0 + np.exp(-35 * (t - knee))))
    return np.cumsum(rate)


def _threshold_rate_shape(rng, t):
    threshold = float(rng.uniform(0.25, 0.75))
    pre_power = float(rng.uniform(0.7, 1.5))
    post_power = float(rng.uniform(1.2, 4.0))
    pre = np.power(np.clip(t / threshold, 0, 1), pre_power) * threshold
    post_t = np.clip((t - threshold) / (1 - threshold), 0, 1)
    post = threshold + (1 - threshold) * np.power(post_t, post_power)
    return np.where(t < threshold, pre, post)


def _cyclic_stress_shape(rng, t):
    base = float(rng.uniform(0.01, 0.25))
    amp = float(rng.uniform(0.2, 1.5))
    freq = float(rng.uniform(1.0, 8.0))
    phase = float(rng.uniform(0, 2 * np.pi))
    stress = 1.0 + amp * (0.5 + 0.5 * np.sin(2 * np.pi * freq * t + phase))

    if rng.random() < 0.5:
        duty = float(rng.uniform(0.15, 0.65))
        square = ((freq * t + phase / (2 * np.pi)) % 1.0 < duty).astype(float)
        stress = stress + float(rng.uniform(0.2, 1.2)) * square

    trend = float(rng.uniform(-0.5, 1.5))
    rate = base + np.maximum(0, stress) * (1 + trend * t)
    return np.cumsum(np.maximum(rate, 1e-6))


def _shock_relaxation_shape(rng, t):
    rate = np.full_like(t, float(rng.uniform(0.005, 0.08)))
    y = np.cumsum(rate)
    n_shocks = int(rng.integers(1, 6))
    for shock_t in rng.uniform(0.1, 0.9, n_shocks):
        magnitude = float(rng.uniform(0.05, 0.35))
        decay = float(rng.uniform(4.0, 25.0))
        jump = (t >= shock_t) * magnitude
        relaxation = (t >= shock_t) * magnitude * float(rng.uniform(0.0, 0.55)) * (1 - np.exp(-decay * (t - shock_t)))
        y = y + jump - relaxation
    return y


def _gamma_process_shape(rng, t):
    length = len(t)
    rate_trend = np.exp(float(rng.uniform(-2.5, 2.0)) * (t - 0.5))
    shape = float(rng.uniform(0.3, 4.0)) * rate_trend
    scale = float(rng.uniform(0.02, 0.25))
    increments = rng.gamma(shape=np.maximum(shape, 1e-4), scale=scale, size=length)
    increments[0] = 0
    return np.cumsum(increments)


def _wiener_drift_shape(rng, t):
    length = len(t)
    drift = float(rng.uniform(0.01, 0.2)) * np.exp(float(rng.uniform(-1.0, 2.0)) * t)
    diffusion = float(rng.uniform(0.005, 0.08))
    increments = drift + rng.normal(0, diffusion, length)
    y = np.cumsum(increments)
    if rng.random() < 0.75:
        y = np.maximum.accumulate(y)
    return y


def _base_degradation_shape(rng, episode_length):
    t = np.linspace(0, 1, episode_length)
    family = rng.choice([
        "power",
        "exponential",
        "gompertz",
        "weibull",
        "saturation",
        "piecewise",
        "threshold",
        "cyclic",
        "shock_relaxation",
        "gamma",
        "wiener",
    ])

    if family == "power":
        p = float(rng.uniform(0.25, 5.0))
        shape = np.power(t, p)
    elif family == "exponential":
        k = float(rng.uniform(0.4, 7.0))
        shape = np.expm1(k * t)
    elif family == "gompertz":
        b = float(rng.uniform(1.5, 8.0))
        c = float(rng.uniform(1.0, 8.0))
        shape = np.exp(-b * np.exp(-c * t))
    elif family == "weibull":
        k = float(rng.uniform(0.4, 4.5))
        lam = float(rng.uniform(0.25, 1.2))
        shape = 1 - np.exp(-np.power(t / lam, k))
    elif family == "saturation":
        k = float(rng.uniform(0.6, 9.0))
        shape = 1 - np.exp(-k * t)
    elif family == "piecewise":
        shape = _piecewise_rate_shape(rng, t)
    elif family == "threshold":
        shape = _threshold_rate_shape(rng, t)
    elif family == "cyclic":
        shape = _cyclic_stress_shape(rng, t)
    elif family == "shock_relaxation":
        shape = _shock_relaxation_shape(rng, t)
    elif family == "gamma":
        shape = _gamma_process_shape(rng, t)
    else:
        shape = _wiener_drift_shape(rng, t)

    shape = _normalize_shape(shape)
    return shape, family


def _compose_degradation_shape(rng, episode_length):
    primary, primary_family = _base_degradation_shape(rng, episode_length)
    if primary is None:
        return None, primary_family

    shape = primary.copy()
    families = [primary_family]
    if rng.random() < 0.55:
        secondary, secondary_family = _base_degradation_shape(rng, episode_length)
        if secondary is not None:
            mix = float(rng.uniform(0.15, 0.55))
            shape = (1 - mix) * shape + mix * secondary
            families.append(secondary_family)

    if rng.random() < 0.35:
        smooth = _random_smooth_signal(rng, episode_length)
        amplitude = float(rng.uniform(0.01, 0.08))
        shape = shape + amplitude * smooth

    return _normalize_shape(shape), "+".join(families)


def _apply_observation_effects(rng, shape):
    observed = np.asarray(shape, dtype=np.float64).copy()
    length = observed.shape[-1]
    progress = np.linspace(0, 1, length)

    if rng.random() < 0.85:
        noise_scale = float(rng.uniform(0.002, 0.035))
        hetero = 0.4 + 1.6 * np.power(progress, float(rng.uniform(0.5, 2.5)))
        observed = observed + noise_scale * hetero * _autocorrelated_noise(rng, length)

    if rng.random() < 0.45:
        drift = float(rng.uniform(-0.06, 0.08))
        observed = observed + drift * np.power(progress, float(rng.uniform(0.7, 2.2)))

    if rng.random() < 0.25:
        n_spikes = int(rng.integers(1, 5))
        spike_idx = rng.integers(1, length, n_spikes)
        observed[spike_idx] += rng.normal(0, float(rng.uniform(0.02, 0.12)), n_spikes)

    if rng.random() < 0.35:
        levels = int(rng.integers(32, 256))
        observed = np.round(observed * levels) / levels

    if rng.random() < 0.75:
        observed = np.maximum.accumulate(observed)

    return _normalize_shape(observed)


def _scale_degradation_episode(rng, shape, max_value=15):
    amplitude = float(rng.uniform(0.3, max_value * 0.75))
    offset = float(rng.uniform(0.0, max(0.01, max_value - amplitude - 0.05)))
    episode = offset + amplitude * shape
    return np.clip(episode, 0, max_value - 1e-5)


def generate_diverse_degradation_episodes(
    episode_length=100,
    n_episodes=5000,
    seed=None,
    max_value=15,
    return_metadata=False,
):
    """
    Generate degradation trajectories from a shape grammar rather than a small set
    of fixed equations.

    The generator intentionally varies normalized curve geometry: acceleration,
    saturation, knees, shocks, cyclic stress, stochastic reliability processes,
    mixed mechanisms, and observation effects. This is useful when training with
    window normalization, where scale-only parameter sweeps collapse to nearly
    identical examples.
    """
    rng = np.random.default_rng(seed)
    episodes = []
    metadata = []
    attempts = 0
    max_attempts = int(n_episodes * 20 + 100)

    while len(episodes) < n_episodes and attempts < max_attempts:
        attempts += 1
        shape, family = _compose_degradation_shape(rng, episode_length)
        if shape is None:
            continue
        observed = _apply_observation_effects(rng, shape)
        if observed is None:
            continue
        episode = _scale_degradation_episode(rng, observed, max_value=max_value)
        if np.isfinite(episode).all() and episode.shape == (episode_length,):
            episodes.append(episode.astype(np.float32))
            metadata.append({"family": family})

    if len(episodes) < n_episodes:
        raise RuntimeError(f"Only generated {len(episodes)} valid episodes after {attempts} attempts.")

    episodes = np.stack(episodes, axis=0)
    if return_metadata:
        return episodes, metadata
    return episodes


def normalized_shape_signatures(episodes, decimals=2):
    episodes = np.asarray(episodes, dtype=np.float64)
    mins = episodes.min(axis=1, keepdims=True)
    spans = episodes.max(axis=1, keepdims=True) - mins
    normalized = (episodes - mins) / (spans + 1e-8)
    return np.round(normalized, decimals=decimals)


def degradation_shape_diagnostics(episodes, signature_decimals=2):
    """
    Summarize shape variety after per-episode normalization.

    This intentionally uses simple NumPy features so it can run in notebooks
    without adding another dependency.
    """
    episodes = np.asarray(episodes, dtype=np.float64)
    if episodes.ndim != 2:
        raise ValueError("episodes must be a 2D array with shape (n_episodes, episode_length).")
    if episodes.shape[1] < 4:
        raise ValueError("episodes need at least 4 time steps for shape diagnostics.")

    mins = episodes.min(axis=1, keepdims=True)
    spans = episodes.max(axis=1, keepdims=True) - mins
    normalized = (episodes - mins) / (spans + 1e-8)
    slopes = np.diff(normalized, axis=1)
    curvature = np.diff(slopes, axis=1)
    signatures = normalized_shape_signatures(episodes, decimals=signature_decimals)
    unique_signatures = np.unique(signatures, axis=0).shape[0]

    early = np.mean(np.abs(slopes[:, : max(1, slopes.shape[1] // 5)]), axis=1)
    late = np.mean(np.abs(slopes[:, -max(1, slopes.shape[1] // 5):]), axis=1)
    final_initial_slope_ratio = late / (early + 1e-8)

    curvature_sign = np.sign(np.where(np.abs(curvature) < 1e-8, 0, curvature))
    curvature_changes = np.sum(curvature_sign[:, 1:] != curvature_sign[:, :-1], axis=1)
    monotonic_fraction = np.mean(np.all(slopes >= -1e-5, axis=1))

    return {
        "n_episodes": int(episodes.shape[0]),
        "episode_length": int(episodes.shape[1]),
        "unique_normalized_signatures": int(unique_signatures),
        "unique_signature_fraction": float(unique_signatures / episodes.shape[0]),
        "monotonic_fraction": float(monotonic_fraction),
        "mean_abs_slope": float(np.mean(np.abs(slopes))),
        "mean_abs_curvature": float(np.mean(np.abs(curvature))),
        "median_final_initial_slope_ratio": float(np.median(final_initial_slope_ratio)),
        "p10_final_initial_slope_ratio": float(np.quantile(final_initial_slope_ratio, 0.10)),
        "p90_final_initial_slope_ratio": float(np.quantile(final_initial_slope_ratio, 0.90)),
        "mean_curvature_sign_changes": float(np.mean(curvature_changes)),
    }
