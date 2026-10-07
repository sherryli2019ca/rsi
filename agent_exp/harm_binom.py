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

which always respects the support rho_k >= -r0.

With slope=True, patched runs instead follow logistic(alpha_t + beta (alpha_t - c)
+ delta_k), c the logit of the mean unpatched failure rate, beta ~ N(0, 1). A
common beta < 0 lets patches move every task's failure rate towards the middle
(raise it on tasks that rarely fail, lower it on tasks that often fail), which
the logit-additive model cannot represent and would otherwise absorb into the
task effects.

Sampling is Metropolis within Gibbs, interweaving centred and non-centred
updates of the patch population: vectorised random-walk updates of every
delta_k and alpha_t (each is conditionally independent of the others of its
kind), conjugate normal updates of a and mu, and random-walk updates of log s_a,
log s_d and beta.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit


def _ll(f, n, eta):
    """Binomial log likelihood without the constant."""
    return f * log_expit(eta) + (n - f) * log_expit(-eta)


def center(n0, f0):
    m = f0.sum() / n0.sum()
    return float(np.log(m / (1 - m)))


def patched(al, d, beta, c):
    """(P, T) patched logits; al (T,) or (S, T) with matching d, beta."""
    return (al * (1 + beta) - beta * c)[..., None, :] + d[..., :, None]


def sample(N, NF, n0, f0, iters=20000, burn=5000, thin=25, seed=0, slope=False):
    """N, NF: (P, T) runs and failures of P patches on T tasks; n0, f0: (T,)
    unpatched runs and failures. Returns dict of draws."""
    rng = np.random.default_rng(seed)
    P, T = N.shape
    p0 = (f0 + 0.5) / (n0 + 1)
    alpha = np.log(p0 / (1 - p0))
    delta = np.zeros(P)
    a, mu, sa, sd, beta = alpha.mean(), 0.0, 1.0, 0.3, 0.0
    c = center(n0, f0)
    step_a, step_d = np.full(T, 0.5), np.full(P, 0.5)
    acc_a, acc_d = np.zeros(T), np.zeros(P)
    out = {k: [] for k in ("delta", "alpha", "mu", "sd", "sa", "a", "beta")}

    def lik_delta(d):
        return _ll(NF, N, patched(alpha, d, beta, c)).sum(1)

    def lik_alpha(al):
        return _ll(f0, n0, al) + _ll(NF, N, patched(al, delta, beta, c)).sum(0)

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
        # common slope (random walk, N(0, 1) prior)
        if slope:
            b_n = beta + 0.05 * rng.standard_normal()
            lr = (_ll(NF, N, patched(alpha, delta, b_n, c)).sum()
                  - _ll(NF, N, patched(alpha, delta, beta, c)).sum() - 0.5 * (b_n ** 2 - beta ** 2))
            if np.log(rng.random()) < lr:
                beta = b_n
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
                         ("sa", sa), ("a", a), ("beta", beta)):
                out[k].append(np.copy(v) if np.ndim(v) else float(v))
    res = {k: np.array(v) for k, v in out.items()}
    res["c"] = np.full(len(res["mu"]), c)
    return res


def rho_draws(draws, wt=None):
    """(S, P) average excess failure probability of each patch over the tasks,
    and (S,) mean unpatched failure probability. wt (T,) weights the tasks: the
    estimand of Eq. 1 averages over previously solved episodes, so a task counts
    in proportion to how often it was solved (the regression-task sampling frame);
    None weights tasks equally."""
    al, de = draws["alpha"], draws["delta"]
    be, c = draws["beta"][:, None], draws["c"][:, None]
    wt = np.full(al.shape[1], 1 / al.shape[1]) if wt is None else wt / wt.sum()
    base = expit(al)
    rho = (expit(patched(al, de, be, c)) - base[:, None, :]) @ wt
    return rho, base @ wt


def new_patch_rho(draws, rng, wt=None):
    """rho of a patch with no regression runs: delta from the fitted population."""
    d = draws["mu"] + draws["sd"] * rng.standard_normal(len(draws["mu"]))
    al, be, c = draws["alpha"], draws["beta"][:, None], draws["c"][:, None]
    wt = np.full(al.shape[1], 1 / al.shape[1]) if wt is None else wt / wt.sum()
    return (expit(al * (1 + be) - be * c + d[:, None]) - expit(al)) @ wt


def _task_stats(f0, n0, NF, N):
    """Patched minus unpatched failure rate on tasks whose unpatched runs never
    failed, and on tasks whose unpatched runs failed at least half the time:
    the task-level pattern a logit-additive model may miss."""
    out = []
    for m in (f0 == 0, f0 >= 0.5 * n0):
        if not m.any() or N[:, m].sum() == 0:
            out.append(np.nan)
            continue
        out.append(NF[:, m].sum() / N[:, m].sum() - f0[m].sum() / n0[m].sum())
    return out


def ppc(draws, N, NF, n0, rng, sel, f0=None):
    """Posterior predictive check: spread of failure rates of patches `sel`,
    number with no failure, maximum failure rate (as in reanalysis3.ppc), and,
    with f0, the task-level excess on rarely and often failing tasks
    (unpatched and patched runs both replicated)."""
    obs_rate = NF[sel].sum(1) / N[sel].sum(1)
    obs = [obs_rate.std(), (NF[sel].sum(1) == 0).sum(), obs_rate.max()]
    names = ["sd", "n_zero", "max"]
    if f0 is not None:
        obs += _task_stats(f0, n0, NF, N)
        names += ["excess_easy", "excess_hard"]
    obs = np.array(obs)
    n0i, Ni = n0.astype(int), N.astype(int)
    sims = []
    for s in range(len(draws["mu"])):
        al = draws["alpha"][s]
        p = expit(patched(al, draws["delta"][s], draws["beta"][s], draws["c"][s]))
        Y = rng.binomial(Ni, p)
        y = Y[sel].sum(1) / N[sel].sum(1)
        row = [y.std(), (y == 0).sum(), y.max()]
        if f0 is not None:
            row += _task_stats(rng.binomial(n0i, expit(al)), n0, Y, N)
        sims.append(row)
    sims = np.array(sims)
    ok = ~np.isnan(sims)
    return {"obs": obs.tolist(), "sim_mean": np.nanmean(sims, 0).tolist(),
            "p_upper": [float((sims[ok[:, i], i] >= obs[i]).mean()) for i in range(len(obs))],
            "stats": names}
