from dataclasses import dataclass
from typing import Callable

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
    r"""
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
    r"""
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
    r"""
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


@dataclass(frozen=True)
class DegradationMechanism:
    name: str
    domain: str
    generator: Callable
    weight: float = 1.0
    source_type: str = "synthetic_mechanistic"


@dataclass(frozen=True)
class CorpusConfig:
    episode_length: int = 100
    n_episodes: int = 5000
    seed: int | None = None
    max_value: float = 15
    source_weights: dict | None = None
    apply_observation_effects: bool = True
    return_metadata: bool = True

    def to_kwargs(self):
        return {
            "episode_length": self.episode_length,
            "n_episodes": self.n_episodes,
            "seed": self.seed,
            "max_value": self.max_value,
            "source_weights": self.source_weights,
            "apply_observation_effects": self.apply_observation_effects,
            "return_metadata": self.return_metadata,
        }


def _metadata(
    name,
    domain,
    observed_variable,
    mechanism_family,
    parameters,
    covariates=None,
    source_type="synthetic_mechanistic",
    monotonic_expected=True,
):
    return {
        "mechanism": name,
        "family": name,
        "domain": domain,
        "source_type": source_type,
        "observed_variable": observed_variable,
        "mechanism_family": mechanism_family,
        "parameters": parameters,
        "covariates": covariates or {},
        "monotonic_expected": bool(monotonic_expected),
    }


def _shape_grammar_mechanism(rng, episode_length):
    shape, family = _compose_degradation_shape(rng, episode_length)
    return shape, _metadata(
        name="shape_grammar",
        domain="generic",
        observed_variable="abstract_degradation",
        mechanism_family=family,
        parameters={"composed_family": family},
        source_type="synthetic_shape_grammar",
        monotonic_expected=False,
    )


def _battery_capacity_fade_mechanism(rng, episode_length):
    t = np.linspace(0, 1, episode_length)
    temperature_c = float(rng.uniform(15, 45))
    c_rate = float(rng.uniform(0.2, 2.5))
    depth_of_discharge = float(rng.uniform(0.2, 1.0))
    arrhenius = np.exp(0.055 * (temperature_c - 25))
    calendar = float(rng.uniform(0.05, 0.45)) * np.sqrt(t + 1e-6) * arrhenius
    cycle = float(rng.uniform(0.05, 0.7)) * np.power(t, float(rng.uniform(0.8, 1.8))) * c_rate * depth_of_discharge
    knee_location = float(rng.uniform(0.55, 0.9))
    knee_severity = float(rng.uniform(0.0, 1.5))
    plating = knee_severity * np.maximum(t - knee_location, 0) ** float(rng.uniform(1.4, 3.5))
    recovery = float(rng.uniform(0.0, 0.025)) * np.sin(2 * np.pi * rng.uniform(2, 10) * t + rng.uniform(0, 2 * np.pi))
    shape = calendar + cycle + plating + recovery
    return _normalize_shape(shape), _metadata(
        name="battery_capacity_fade",
        domain="battery",
        observed_variable="capacity_loss",
        mechanism_family="calendar_plus_cycle_aging_with_optional_knee",
        parameters={"knee_location": knee_location, "knee_severity": knee_severity},
        covariates={"temperature_c": temperature_c, "c_rate": c_rate, "depth_of_discharge": depth_of_discharge},
    )


def _fatigue_crack_growth_mechanism(rng, episode_length):
    stress = _random_smooth_signal(rng, episode_length, anchors=int(rng.integers(5, 14)))
    stress = np.clip(0.6 + 0.25 * stress + rng.uniform(0.0, 0.6), 0.05, None)
    a = np.zeros(episode_length)
    a[0] = float(rng.uniform(0.02, 0.12))
    c = float(10 ** rng.uniform(-3.2, -1.2))
    m = float(rng.uniform(2.0, 4.5))
    threshold = float(rng.uniform(0.02, 0.18))
    for i in range(episode_length - 1):
        delta_k = stress[i] * np.sqrt(max(a[i], 1e-6))
        growth = c * max(delta_k - threshold, 0) ** m
        if rng.random() < 0.015:
            growth += float(rng.uniform(0.002, 0.04))
        a[i + 1] = a[i] + growth
    return _normalize_shape(a), _metadata(
        name="fatigue_crack_growth",
        domain="fatigue",
        observed_variable="crack_length",
        mechanism_family="paris_law_with_variable_amplitude_loading",
        parameters={"c": c, "m": m, "threshold": threshold},
        covariates={"stress_mean": float(stress.mean()), "stress_std": float(stress.std())},
    )


def _creep_deformation_mechanism(rng, episode_length):
    t = np.linspace(0, 1, episode_length)
    primary_exp = float(rng.uniform(0.25, 0.7))
    secondary_rate = float(rng.uniform(0.05, 0.45))
    tertiary_start = float(rng.uniform(0.55, 0.88))
    tertiary_power = float(rng.uniform(1.8, 4.5))
    stress_fraction = float(rng.uniform(0.35, 0.95))
    temperature_fraction = float(rng.uniform(0.35, 0.95))
    primary = float(rng.uniform(0.15, 0.6)) * (1 - np.exp(-float(rng.uniform(3, 12)) * t)) ** primary_exp
    secondary = secondary_rate * t
    tertiary = float(rng.uniform(0.2, 1.8)) * np.maximum(t - tertiary_start, 0) ** tertiary_power
    shape = primary + secondary + tertiary
    return _normalize_shape(shape), _metadata(
        name="creep_deformation",
        domain="materials",
        observed_variable="creep_strain",
        mechanism_family="primary_secondary_tertiary_creep",
        parameters={"tertiary_start": tertiary_start, "tertiary_power": tertiary_power},
        covariates={"stress_fraction": stress_fraction, "temperature_fraction": temperature_fraction},
    )


def _corrosion_pitting_mechanism(rng, episode_length):
    t = np.linspace(0, 1, episode_length)
    humidity = float(rng.uniform(0.35, 1.0))
    chloride = float(rng.uniform(0.0, 1.0))
    passivation_strength = float(rng.uniform(0.0, 0.7))
    uniform = float(rng.uniform(0.02, 0.5)) * humidity * t
    passivation = passivation_strength * (1 - np.exp(-float(rng.uniform(2, 8)) * t))
    n_pits = int(rng.integers(1, 8))
    pit_depth = np.zeros_like(t)
    for onset in rng.uniform(0.05, 0.9, n_pits):
        severity = float(rng.uniform(0.03, 0.5)) * (0.4 + chloride)
        power = float(rng.uniform(0.8, 2.5))
        pit_depth += severity * np.maximum(t - onset, 0) ** power
    shape = uniform + pit_depth - 0.15 * passivation
    return _normalize_shape(np.maximum.accumulate(shape)), _metadata(
        name="corrosion_pitting",
        domain="corrosion",
        observed_variable="max_pit_depth",
        mechanism_family="uniform_corrosion_plus_stochastic_pit_initiation",
        parameters={"n_pits": n_pits, "passivation_strength": passivation_strength},
        covariates={"humidity": humidity, "chloride": chloride},
    )


def _wear_transition_mechanism(rng, episode_length):
    t = np.linspace(0, 1, episode_length)
    load = float(rng.uniform(0.2, 1.0))
    lubrication_quality = float(rng.uniform(0.0, 1.0))
    running_in = float(rng.uniform(0.1, 0.7)) * (1 - np.exp(-float(rng.uniform(6, 18)) * t))
    steady = float(rng.uniform(0.03, 0.5)) * load * t * (1.2 - 0.7 * lubrication_quality)
    transition = float(rng.uniform(0.45, 0.9))
    severe = float(rng.uniform(0.0, 1.8)) * np.maximum(t - transition, 0) ** float(rng.uniform(1.4, 3.2))
    stick_slip = float(rng.uniform(0.0, 0.05)) * np.maximum(0, np.sin(2 * np.pi * rng.uniform(4, 16) * t + rng.uniform(0, 2 * np.pi)))
    shape = running_in + steady + severe + stick_slip
    return _normalize_shape(shape), _metadata(
        name="wear_transition",
        domain="wear",
        observed_variable="wear_depth",
        mechanism_family="running_in_steady_wear_severe_wear_transition",
        parameters={"transition": transition},
        covariates={"load": load, "lubrication_quality": lubrication_quality},
    )


def default_degradation_mechanism_registry():
    return [
        DegradationMechanism("shape_grammar", "generic", _shape_grammar_mechanism, weight=1.5, source_type="synthetic_shape_grammar"),
        DegradationMechanism("battery_capacity_fade", "battery", _battery_capacity_fade_mechanism, weight=1.0),
        DegradationMechanism("fatigue_crack_growth", "fatigue", _fatigue_crack_growth_mechanism, weight=1.0),
        DegradationMechanism("creep_deformation", "materials", _creep_deformation_mechanism, weight=1.0),
        DegradationMechanism("corrosion_pitting", "corrosion", _corrosion_pitting_mechanism, weight=1.0),
        DegradationMechanism("wear_transition", "wear", _wear_transition_mechanism, weight=1.0),
    ]


def _select_mechanism(rng, registry, source_weights=None):
    names = [m.name for m in registry]
    weights = np.array([m.weight for m in registry], dtype=np.float64)
    if source_weights:
        weights = np.array(
            [source_weights.get(name, source_weights.get(m.domain, weight)) for name, m, weight in zip(names, registry, weights)],
            dtype=np.float64,
        )
    weights = np.clip(weights, 0, None)
    if weights.sum() <= 0:
        raise ValueError("At least one mechanism weight must be positive.")
    probs = weights / weights.sum()
    return registry[int(rng.choice(len(registry), p=probs))]


def generate_degradation_corpus(
    episode_length=100,
    n_episodes=5000,
    seed=None,
    max_value=15,
    mechanisms=None,
    source_weights=None,
    apply_observation_effects=True,
    return_metadata=True,
):
    """
    Generate a structured degradation corpus from a registry of mechanisms.

    Each mechanism returns a normalized latent degradation shape plus metadata
    describing the domain, observed variable, sampled parameters, and covariates.
    The corpus generator then applies observation effects and rescales the result
    into the numeric range used by the downstream digitizer.
    """
    rng = np.random.default_rng(seed)
    registry = mechanisms or default_degradation_mechanism_registry()
    episodes = []
    metadata = []
    attempts = 0
    max_attempts = int(n_episodes * 25 + 100)

    while len(episodes) < n_episodes and attempts < max_attempts:
        attempts += 1
        mechanism = _select_mechanism(rng, registry, source_weights=source_weights)
        shape, item_metadata = mechanism.generator(rng, episode_length)
        if shape is None:
            continue
        observed = _apply_observation_effects(rng, shape) if apply_observation_effects else _normalize_shape(shape)
        if observed is None:
            continue
        episode = _scale_degradation_episode(rng, observed, max_value=max_value)
        if np.isfinite(episode).all() and episode.shape == (episode_length,):
            item_metadata = dict(item_metadata)
            item_metadata.setdefault("source_type", mechanism.source_type)
            item_metadata["registry_mechanism"] = mechanism.name
            item_metadata["registry_domain"] = mechanism.domain
            item_metadata["episode_length"] = int(episode_length)
            item_metadata["max_value"] = float(max_value)
            item_metadata["observation_effects"] = bool(apply_observation_effects)
            episodes.append(episode.astype(np.float32))
            metadata.append(item_metadata)

    if len(episodes) < n_episodes:
        raise RuntimeError(f"Only generated {len(episodes)} valid episodes after {attempts} attempts.")

    episodes = np.stack(episodes, axis=0)
    if return_metadata:
        return episodes, metadata
    return episodes


def generate_degradation_corpus_from_config(config, mechanisms=None):
    return generate_degradation_corpus(**config.to_kwargs(), mechanisms=mechanisms)


def corpus_metadata_summary(metadata):
    summary = {"n_episodes": len(metadata), "mechanisms": {}, "domains": {}, "source_types": {}}
    for item in metadata:
        for key, bucket in (("mechanism", "mechanisms"), ("domain", "domains"), ("source_type", "source_types")):
            value = item.get(key, "unknown")
            summary[bucket][value] = summary[bucket].get(value, 0) + 1
    return summary


def corpus_composition_table(metadata, key="mechanism"):
    counts = {}
    for item in metadata:
        value = item.get(key, "unknown")
        counts[value] = counts.get(value, 0) + 1

    total = max(1, len(metadata))
    rows = [
        {"value": value, "count": int(count), "fraction": float(count / total)}
        for value, count in counts.items()
    ]
    return sorted(rows, key=lambda row: (-row["count"], row["value"]))


def corpus_shape_diagnostics_by_group(
    episodes,
    metadata,
    group_key="mechanism",
    signature_decimals=2,
):
    episodes = np.asarray(episodes)
    if len(metadata) != episodes.shape[0]:
        raise ValueError("metadata length must match the number of episodes.")

    groups = {}
    for idx, item in enumerate(metadata):
        groups.setdefault(item.get(group_key, "unknown"), []).append(idx)

    diagnostics = {}
    for group, indices in groups.items():
        group_episodes = episodes[np.asarray(indices, dtype=int)]
        group_diagnostics = degradation_shape_diagnostics(
            group_episodes,
            signature_decimals=signature_decimals,
        )
        group_diagnostics["fraction"] = float(len(indices) / max(1, episodes.shape[0]))
        diagnostics[group] = group_diagnostics
    return diagnostics


def corpus_window_budget_summary(
    episode_length,
    context_window,
    future_window=1,
    stride=1,
    n_episodes=None,
):
    trainable_span = int(episode_length) - int(context_window) - int(future_window) + 1
    windows_per_episode = max(0, (trainable_span + int(stride) - 1) // int(stride))
    summary = {
        "episode_length": int(episode_length),
        "context_window": int(context_window),
        "future_window": int(future_window),
        "stride": int(stride),
        "windows_per_episode": int(windows_per_episode),
    }
    if n_episodes is not None:
        summary["n_episodes"] = int(n_episodes)
        summary["total_windows"] = int(windows_per_episode * int(n_episodes))
    return summary


def corpus_diagnostics_report(
    episodes,
    metadata,
    group_key="mechanism",
    signature_decimals=2,
    context_window=None,
    future_window=1,
    stride=1,
):
    report = {
        "overall_shape": degradation_shape_diagnostics(
            episodes,
            signature_decimals=signature_decimals,
        ),
        "mechanism_table": corpus_composition_table(metadata, key="mechanism"),
        "domain_table": corpus_composition_table(metadata, key="domain"),
        "source_type_table": corpus_composition_table(metadata, key="source_type"),
        "shape_by_group": corpus_shape_diagnostics_by_group(
            episodes,
            metadata,
            group_key=group_key,
            signature_decimals=signature_decimals,
        ),
    }
    if context_window is not None:
        report["window_budget"] = corpus_window_budget_summary(
            episode_length=np.asarray(episodes).shape[1],
            context_window=context_window,
            future_window=future_window,
            stride=stride,
            n_episodes=np.asarray(episodes).shape[0],
        )
    return report


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
    return generate_degradation_corpus(
        episode_length=episode_length,
        n_episodes=n_episodes,
        seed=seed,
        max_value=max_value,
        return_metadata=return_metadata,
    )


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
