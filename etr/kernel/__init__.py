"""Blind blur-kernel estimation (the blur module).

    synth       synthetic blur + fixed evaluation kernel sets
    signature   the blur's phase-blind signature (gradient autocorrelation)
    heads       signature head: signature -> K = 32 kernel hypotheses in the kernel VAE's latent; retrieval embedding
    restore     Wiener restore, reblur, flip score, decoding
    critic      learned critic: scores a candidate by its restoration's artefacts
    estimator   KernelEstimator: candidate pool (crops x 8 views, bank, perturbations, safe set) -> pick
    bank        the retrieval bank's kernels, regenerated from (size, seed)
    linearise   the linearised (blur-stage) observation
"""
