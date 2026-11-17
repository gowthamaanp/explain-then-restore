"""Learned networks.

    prior          PriorNet: NAFNet-style U-Net backbone of the Prior
    gs_denoiser    GSDenoiser: gradient-step denoiser D = I - grad g built on PriorNet (the Prior)
    goal           GoalEncoder: the conditioning vector of the Prior (fidelity preset)
    kernel_prior   kernel prior C: spatial VAE over blur kernels
    blocks         NAFBlock, DWConv3x3, LayerNorm2d, FiLM, Down, Up
"""
