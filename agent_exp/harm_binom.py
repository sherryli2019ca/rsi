"""Task-level binomial hierarchical harm model (fifth review).

Regression outcomes are modelled per patch and task, with the unpatched
baseline runs on the same tasks:

    f0_t ~ Bin(n0_t, logistic(alpha_t))                 unpatched runs of task t
    f_kt ~ Bin(n_kt, logistic(alpha_t + delta_k))       runs of patch k on task t
    alpha_t ~ N(a, s_a^2),  delta_k ~ N(mu, s_d^2)
    a, mu ~ N(0, 3^2),  s_a ~ HalfNormal(2),  s_d ~ HalfNormal(1)

Task effects alpha_t are shared by the baseline and every patch, so tasks that
fail often unpatched no longer look like harmful patches, and the uncertainty
in the between-patch spread s_d is carried into every draw. A patch's harm is
its average excess failure probability over the regression tasks,

    rho_k = mean_t [logistic(alpha_t + delta_k) - logistic(alpha_t)],

which always respects the support rho_k >= -r0. Sampling is Metropolis within
Gibbs, interweaving centred and non-centred updates of the patch population: vectorised random-walk updates of every delta_k and alpha_t (each is
conditionally independent of the others of its kind), conjugate normal updates
of a and mu, and random-walk updates of log s_a and log s_d.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit


def _ll(f, n, eta):
    """Binomial log likelihood without the constant."""
    return f * log_expit(eta) + (n - f) * log_expit(-eta)


def sample(N, NF, n0, f0, iters=20000, burn=5000, thin=25, seed=0):
    """N, NF: (P, T) runs and failures of P patches on T tasks; n0, f0: (T,)
    unpatched runs and failures. Returns dict of draws."""
    rng = np.random.default_rng(seed)
    P, T = N.shape
    p0 = (f0 + 0.5) / (n0 + 1)
    alpha = np.log(p0 / (1 - p0))
    delta = np.zeros(P)
    a, mu, sa, sd = alpha.mean(), 0.0, 1.0, 0.3
    step_a, step_d = np.full(T, 0.5), np.full(P, 0.5)
    acc_a, acc_d = np.zeros(T), np.zeros(P)
    out = {k: [] for k in ("delta", "alpha", "mu", "sd", "sa", "a")}

    def lik_delta(d):
        return _ll(NF, N, alpha[None, :] + d[:, None]).sum(1)

    def lik_alpha(al):
        return _ll(f0, n0, al) + _ll(NF, N, al[None, :] + delta[:, None]).sum(0)

    for it in range(iters):
        # patch effects
        prop = delta + step_d * rng.standard_normal(P)
        lr = (lik_delta(prop) - lik_delta(delta)
              - 0.5 * ((prop - mu) ** 2 - (delta - mu) ** 2) / sd ** 2)
        ok = np.log(rng.random(P)) < lr
        delta = np.where(ok, prop, delta)
        acc_d += ok
        # task effects
        prop = alpha + step_a * rng.standard_normal(T)
        lr = (lik_alpha(prop) - lik_alpha(alpha)
              - 0.5 * ((prop - a) ** 2 - (alpha - a) ** 2) / sa ** 2)
        ok = np.log(rng.random(T)) < lr
        alpha = np.where(ok, prop, alpha)
        acc_a += ok
        # population means (conjugate, N(0, 3^2) priors)
        for name, x, s in (("mu", delta, sd), ("a", alpha, sa)):
            prec = len(x) / s ** 2 + 1 / 9
            m = (x.sum() / s ** 2) / prec
            v = m + rng.standard_normal() / np.sqrt(prec)
            if name == "mu":
                mu = v
            else:
                a = v
        # population spreads (half-normal priors, random walk on the log scale)
        for name, x, m, scale in (("sd", delta, mu, 1.0), ("sa", alpha, a, 2.0)):
            cur = sd if name == "sd" else sa
            new = cur * np.exp(0.2 * rng.standard_normal())

            def lp(s):
                return (-len(x) * np.log(s) - 0.5 * ((x - m) ** 2).sum() / s ** 2
                        - 0.5 * (s / scale) ** 2 + np.log(s))   # + log s: Jacobian
            if np.log(rng.random()) < lp(new) - lp(cur):
                if name == "sd":
                    sd = new
                else:
                    sa = new
        # interweaving step (non-centred): move mu and s_d with the standardised
        # patch effects z = (delta - mu) / s_d held fixed, which mixes s_d where
        # the centred updates above stall (small s_d)
        z = (delta - mu) / sd
        mu_n = mu + 0.05 * rng.standard_normal()
        sd_n = sd * np.exp(0.2 * rng.standard_normal())
        d_n = mu_n + sd_n * z
        lr = (lik_delta(d_n).sum() - lik_delta(delta).sum()
              - (mu_n ** 2 - mu ** 2) / 18 - 0.5 * (sd_n ** 2 - sd ** 2)
              + np.log(sd_n / sd))
        if np.log(rng.random()) < lr:
            mu, sd, delta = mu_n, sd_n, d_n
        # adapt step sizes during burn-in towards ~40% acceptance
        if it < burn and it % 100 == 99:
            step_d *= np.exp((acc_d / 100 - 0.4))
            step_a *= np.exp((acc_a / 100 - 0.4))
            acc_d[:], acc_a[:] = 0, 0
        if it >= burn and (it - burn) % thin == 0:
            for k, v in (("delta", delta), ("alpha", alpha), ("mu", mu), ("sd", sd),
                         ("sa", sa), ("a", a)):
                out[k].append(np.copy(v) if np.ndim(v) else float(v))
    return {k: np.array(v) for k, v in out.items()}


def rho_draws(draws):
    """(S, P) average excess failure probability of each patch over the tasks,
    and (S,) mean unpatched failure probability."""
    al, de = draws["alpha"], draws["delta"]
    base = expit(al)
    rho = (expit(al[:, None, :] + de[:, :, None]) - base[:, None, :]).mean(-1)
    return rho, base.mean(-1)


def new_patch_rho(draws, rng):
    """rho of a patch with no regression runs: delta from the fitted population."""
    d = draws["mu"] + draws["sd"] * rng.standard_normal(len(draws["mu"]))
    al = draws["alpha"]
    return (expit(al + d[:, None]) - expit(al)).mean(-1)


def ppc(draws, N, NF, n0, rng, sel):
    """Posterior predictive check on patches `sel`: spread of failure rates,
    number with no failure, maximum failure rate (as in reanalysis3.ppc)."""
    obs_rate = NF[sel].sum(1) / N[sel].sum(1)
    obs = np.array([obs_rate.std(), (NF[sel].sum(1) == 0).sum(), obs_rate.max()])
    sims = []
    for s in range(len(draws["mu"])):
        p = expit(draws["alpha"][s][None, :] + draws["delta"][s][sel][:, None])
        y = rng.binomial(N[sel].astype(int), p).sum(1) / N[sel].sum(1)
        sims.append([y.std(), (y == 0).sum(), y.max()])
    sims = np.array(sims)
    return {"obs": obs.tolist(), "sim_mean": sims.mean(0).tolist(),
            "p_upper": (sims >= obs).mean(0).tolist(), "stats": ["sd", "n_zero", "max"]}
