# SPD-License-Identifier: MIT

using Knet
using CUDA

include("layers.jl")

#======================================================================
Model definition.
======================================================================#

function buildmodel(config, indim, outdim)
    dropout = config["dropout"]
    dilations = config["dilations"]
    hdims = config["hidden-dims"]
    kernelsizes = config["kernel-sizes"]
    strides = config["strides"]

    @assert (length(kernelsizes) == length(dilations)
             && length(dilations) == length(strides)
             && length(dilations) == length(hdims))

    layers = Any[]
    push!(layers, PermuteDims([2, 1, 3]))
    push!(layers, AddExtraDim()) # Necessary as convolution is 2D.
    for i in 1:length(hdims)
        layer = Conv(
            (kernelsizes[i], 1),
            indim => hdims[i],
            stride = (strides[i], 1),
            dilation = (dilations[i], 1),
            padding = ((kernelsizes[i] - 1 ) ÷ 2, 0)
           )
        push!(layers, layer)
        push!(layers, BatchNorm(hdims[i], relu))
        push!(layers, Dropout(dropout))
        indim = hdims[i]
    end

    push!(layers, RemoveExtraDim())
    push!(layers, PermuteDims([2, 1, 3]))

    # Final layer is just an affine transform.
    push!(layers, Dense(hdims[end], outdim))

    network = Chain(layers...)
end

function getlengths(config, inlengths)
    dilations = config["dilations"]
    hdims = config["hidden-dims"]
    kernelsizes = config["kernel-sizes"]
    strides = config["strides"]

    seqlengths = inlengths
    for i in 1:length(dilations)
        ksize = kernelsizes[i]
        pad = (ksize - 1 ) ÷ 2
        newlengths = seqlengths .+ 2*pad .- dilations[i]*(ksize-1) .+ strides[i] .- 1
        seqlengths = newlengths .÷ strides[i]
    end
    seqlengths
end

#======================================================================
Training utilities.
======================================================================#

mutable struct PlateauScheduler
    factor
    patience
    threshold
    best_loss
    nsteps
end

function getscheduler(trainconfig)
    sconf = trainconfig["scheduler"]
    PlateauScheduler(sconf["factor"], sconf["patience"], 1e-4, -Inf, 0)
end

function update_scheduler!(s::PlateauScheduler, opts, loss)
    if loss < s.best_loss * (1 - s.threshold)
        s.nsteps = 0
        s.best_loss = loss
    elseif s.nsteps > s.patience
        @debug "$(s.nsteps) epoch(s) without improvement, setting learning rate to: $(opt.eta)"
        for opt in opts opt.lr = opt.lr * s.factor end
        s.nsteps = 0
    end
    s.nsteps += 1
end

function getoptimizer(trainconfig)
    Adam(lr = trainconfig["learning_rate"], beta1 = 0.9, beta2 = 0.999)
end

