"""Blind inference: an observation y -> an estimated, ordered degradation chain theta_hat.

The chain estimator (ProposalNet, etr/world/proposal.py) estimates which factors are present, their parameters and
three order flags; the blur module (etr/kernel) estimates the kernel shape from the linearised observation
(etr/kernel/linearise.py). This module turns those estimates into an ordered list of `Step`s that the data term
(etr/world/chain.py) accepts.

Order. The estimator predicts only three order flags (tone before every affine step; noise before resampling; noise
before tone). Affine steps are placed in the physical order of image formation: scene (haze) -> optics (blur) ->
sensor sampling (resample), and the rain layer after them on the observation grid. JPEG is always last. If the
resulting order is not representable (a tone curve between an affine step and JPEG), the tone curve is moved to the
front.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from etr.world.chain import Step, parse
from etr.world.generator import Q8
from etr.world.proposal import KINDS, SPACES, ProposalNet, monotone_knots
from etr.world.proposal_consts import RS_FAMILIES, LANCZOS_N
from etr.world.resamplers import resampler_psf

DEFAULT_KERR = 2.0 / 255      # kernel-error allowance for an estimated kernel (the config sets it)


def load_proposal(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ProposalNet(n_hyp=ck.get("n_hyp", 0), tone_hist=any(k.startswith("tone_mlp.") for k in ck["model"]),
                       rain=any(k.startswith("rain_head.") for k in ck["model"]),
                       tone_v2=any(k.startswith("tone_v2.") for k in ck["model"])).to(device)
    net.load_state_dict(ck["model"]); net.eval().requires_grad_(False)
    return net


def _T(img, t):
    r, tr = t % 4, t >= 4
    if tr:
        img = img.transpose(-2, -1)
    return torch.rot90(img, r, (-2, -1))


def _Tinv(img, t):
    r, tr = t % 4, t >= 4
    img = torch.rot90(img, -r, (-2, -1))
    return img.transpose(-2, -1) if tr else img


SPATIAL = ("t_mu", "t_lv", "rain")


def tta_forward(net, y, s, views=range(8)):
    """The chain estimator averaged over the 8 dihedral views of y (spatial maps mapped back first). Degradation
    parameters are invariant to these transforms except the blur shape, which the estimator does not output."""
    acc = None
    for t in views:
        o = net(_T(y, t), s)
        o = {k: (_Tinv(v, t) if k in SPATIAL else v) for k, v in o.items()}
        acc = o if acc is None else {k: acc[k] + o[k] for k in acc}
    n = len(list(views))
    return {k: v / n for k, v in acc.items()}


def estimate_theta(net, y, s, thresholds=None, tta=False):
    """Raw estimates (python values / tensors) from ProposalNet for one image.
    thresholds: per-factor logit thresholds (default 0 for every factor)."""
    o = tta_forward(net, y, s) if tta else net(y, s)
    thr = torch.tensor([(thresholds or {}).get(k, 0.0) for k in KINDS], device=y.device)
    p = (o["presence"][0] > thr).tolist()
    th = dict(present=[k for k, on in zip(KINDS, p) if on], presence_logits=o["presence"][0].tolist())
    th["resample"] = dict(family=RS_FAMILIES[int(o["rs_family"][0].argmax())], antialias=bool(int(o["rs_aa"][0].argmax())),
                          a=float(o["rs_a_mu"][0]), n=LANCZOS_N[int(o["rs_n"][0].argmax())],
                          sigma=float(o["rs_sig_mu"][0]))
    nt = int(o["noise_type"][0].argmax())
    mu = o["noise_mu"][0]
    th["noise"] = (dict(type="gauss", sigma=float(mu[0].exp())) if nt == 0 else
                   dict(type="pg", alpha=float(mu[1].exp()), beta=float(mu[2].exp())))
    th["jpeg"] = dict(q=int(round(float(o["q_mu"][0]) * 100)), subsampling="420" if int(o["subsampling"][0].argmax()) == 0 else "444")
    th["tone"] = dict(kind="spline", knots_y=[float(v) for v in monotone_knots(o["knots_mu"][0])], space=SPACES[int(o["space"][0].argmax())])
    if "u_mu" in o:                                   # tone v2: inverse curve u at the observed V quantiles q
        q, u = o["tone_q"][0], o["u_mu"][0]
        p_t = float(torch.sigmoid(o["presence"][0][KINDS.index("tone")]))
        th["tone_unhedged"] = dict(kind="pwl", kx=[float(v) for v in torch.cummax(u, 0).values + 1e-5 * torch.arange(len(u), device=u.device)],
                                   ky=[float(v) for v in q], space=SPACES[int(o["space"][0].argmax())])
        u = p_t * u + (1 - p_t) * q                    # hedge toward the identity by the presence belief
        u = torch.cummax(u, 0).values + 1e-5 * torch.arange(len(u), device=u.device)
        th["tone"] = dict(kind="pwl", kx=[float(v) for v in u], ky=[float(v) for v in q], space=th["tone"]["space"],
                          spread=float((o["tone_u"][0] - o["u_mu"][0][None]).abs().mean()))
    th["haze"] = dict(A=float(o["A_mu"][0]), t=o["t_mu"])                                     # t: [1,1,h,w]
    if "rain" in o:
        th["rain"] = o["rain"]                                                                 # [1,1,h,w] layer estimate
    th["order"] = (o["order"][0] > 0).tolist()                                                # tone_first, noise<resample, noise<tone
    return th


def build_chain(th, y, s, kernel=None, kerr=DEFAULT_KERR):
    """theta estimates -> list of Steps. kernel: [79,79] blur estimate (needed if blur present)."""
    pres = set(th["present"]) - {"layer"}
    rain = "layer" in th["present"] and th.get("rain") is not None
    if s > 1:
        pres.add("resample")              # the requested output size says the image was downsampled
    else:
        pres.discard("resample")
    tone_first, noise_before_rs, noise_before_tone = th["order"]
    H, W = y.shape[-2:]
    hr = (H * s, W * s)
    lin = []
    if "haze" in pres:
        t = th["haze"]["t"].clamp(0.05, 1.0)
        if s > 1:
            t = F.interpolate(t, size=hr, mode="bilinear", align_corners=False)
        lin.append(Step("haze", dict(t=t, A=th["haze"]["A"]), Q8))
    if "blur" in pres:
        assert kernel is not None, "blur present: pass the blur module's estimate"
        lin.append(Step("blur", dict(k=kernel.to(y.device)), Q8 + kerr ** 2))
    if "resample" in pres:
        r = th["resample"]
        k = resampler_psf(s, r["family"], r["antialias"] or r["family"] == "gauss", r["a"], r["n"], r["sigma"], device=y.device)
        lin.append(Step("resample", dict(s=s, family=r["family"], antialias=r["antialias"], k=k), Q8))

    if rain:                                           # additive layer at the observation grid, after the linear factors
        L = th["rain"].clamp(0, 1)
        lin.append(Step("layer", dict(L=L.expand(1, 3, -1, -1).contiguous(), M=(y >= 0.996).float()), Q8))

    def noise_step(at_hr):
        n = th["noise"]
        if n["type"] == "gauss":
            return Step("noise", dict(type="gauss", sigma=n["sigma"], var=n["sigma"] ** 2), Q8)
        ref = F.interpolate(y, size=hr, mode="bicubic", align_corners=False) if at_hr else y
        # a FLAT map at the image-mean variance: alpha * y + beta^2 would carry y's own noise into the data-term
        # weights, which costs several dB on Poisson-Gaussian images
        v = float((n["alpha"] * ref.clamp(0, 1) + n["beta"] ** 2).mean())
        return Step("noise", dict(type="pg", alpha=n["alpha"], beta=n["beta"], var=torch.full_like(ref, v)), Q8)

    steps = list(lin)
    trailing = []
    if "noise" in pres:
        if "resample" in pres and noise_before_rs:
            i = [q.kind for q in steps].index("resample")
            steps.insert(i, noise_step(True))
        else:
            trailing.append(noise_step(False))
    tone = Step("tone", dict(th["tone"]), Q8) if "tone" in pres else None
    if tone is not None:
        if tone_first or "jpeg" in pres or not lin:
            steps.insert(0, tone)
        elif trailing and noise_before_tone:
            trailing.append(tone)
        else:
            trailing.insert(0, tone)
    steps += trailing
    if "jpeg" in pres:
        steps.append(Step("jpeg", dict(th["jpeg"])))
    try:
        parse(steps)
    except NotImplementedError:                       # move the tone curve to the front, else drop it
        steps = [q for q in steps if q.kind != "tone"]
        if tone is not None:
            try:
                parse([tone] + steps)
                steps = [tone] + steps
            except NotImplementedError:
                pass
    return steps


DEFAULTS = dict(tta=False, trust_power=0.0, kerr=DEFAULT_KERR, tau_blur=-1e9, tau_rain=0.0, thresholds=None)


def blind_chain(net, est, y, s, cfg=None):
    """y (+ the scale s implied by the requested output size) -> (steps, theta).
    cfg (configs/etr.json "inference"): tta (dihedral averaging of the chain estimator), trust_power (frequency trust of
    the kernel, 0 = off), kerr (kernel-error standard deviation), tau_blur (minimum kernel margin over no blur; below it
    the blur step is dropped), tau_rain (minimum rain presence probability), thresholds (per-factor detection logits)."""
    from etr.kernel.linearise import lin_from_theta, linearise
    from etr.kernel.scoring import trust_kernel
    c = dict(DEFAULTS, **(cfg or {}))
    th = estimate_theta(net, y, s, c["thresholds"], tta=c["tta"])
    th["fallback"] = []
    if "layer" in th["present"]:
        p_rain = float(torch.sigmoid(torch.tensor(th["presence_logits"][KINDS.index("layer")])))
        if p_rain < c["tau_rain"]:
            th["present"] = [k for k in th["present"] if k != "layer"]; th["fallback"].append("rain")
    kernel = None
    if "blur" in th["present"]:
        y_lin, sig = linearise(y, lin_from_theta(th, y, s))
        kernel, info = est.estimate(y_lin, float(sig))
        th["kernel_info"] = {k: v for k, v in info.items() if not torch.is_tensor(v)}
        if info["is_delta"] or info["total_margin"] < c["tau_blur"]:
            th["present"] = [k for k in th["present"] if k != "blur"]; th["fallback"].append("blur")
            kernel = None
        elif c["trust_power"] > 0:
            kernel = trust_kernel(kernel, info["top"], info["top_w"], c["trust_power"])
    return build_chain(th, y, s, kernel, c["kerr"]), th
