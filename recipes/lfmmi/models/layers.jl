# SPD-License-Identifier: MIT
#
# This is a thin wrapper over the Knet package to have a more high-level
# API to construct the neural network.

using CUDA
using Knet

const DEFAULT_ARRAY_TYPE = Array{Float32}

trainmode!(model) = model
testmode!(model) = model
reallocate(obj, atype) = deepcopy(obj)

gpu(obj) = reallocate(obj, CuArray{Float32})
cpu(obj) = reallocate(obj, Array{Float32})

"""
    permutedims(perm)

apply a permutation. `perm` is an array of the indices of the
dimension, e.g. `perm = [2, 1, 3]` will permute dimension 2 and 1.
"""
struct PermuteDims
    perm
end
(p::PermuteDims)(X) = permutedims(X, p.perm)

"""
    AddExtraDim(perm)

Add an extra second dimension. This is usually to use 2d-convolution on
1d signals.
"""
struct AddExtraDim end
(::AddExtraDim)(X) = reshape(X, size(X,1), 1, size(X,2), size(X,3))

"""
    RemoveExtraDim(perm)

Remove the extra second dimension. See [`AddExtraDim`](@ref).
"""
struct RemoveExtraDim end
(::RemoveExtraDim)(X) = reshape(X, size(X,1), size(X,3), size(X,4))

"""
    Dropout(pdrop)

Dropout layer. `pdrop` is the probability of a value to be set to zero.
"""
mutable struct Dropout
    pdrop
    training
end
(d::Dropout)(X) = dropout(X, d.pdrop; drop = d.training)
Dropout(pdrop) = Dropout(pdrop, true)
trainmode!(d::Dropout) = d.training = true
testmode!(d::Dropout) = d.training = false

"""
    Dense(in, out, σ = identity)

Dense layer.
"""
struct Dense
    W
    b
    σ
end
function (d::Dense)(X)
    rX = reshape(X, size(X,1), :)
    Y = d.σ.( d.W * rX .+ d.b )
    reshape(Y, :, size(X)[2:end]...)
end

Dense(in::Int, out::Int, σ = identity) =
    Dense(param(out, in, atype = DEFAULT_ARRAY_TYPE),
          param0(out, atype = DEFAULT_ARRAY_TYPE), σ)

reallocate(d::Dense, atype) =
        Dense(Param(atype(value(d.W))), Param(atype(value(d.b))), d.σ)

"""
    Conv(kernelsize, inchannels => outchannels; stride = (1,1), padding = (0,0), dilation=(1,1))

2D-convolution layer.
"""
struct Conv
    W
    b
    stride
    padding
    dilation
end

function Conv(ksize, in_out; stride = 1, padding = 0, dilation = 1)
    W = param(ksize..., in_out.first, in_out.second, atype = DEFAULT_ARRAY_TYPE)
    b = param0(1, 1, in_out.second, 1, atype = DEFAULT_ARRAY_TYPE)
    Conv(W, b, stride, padding, dilation)
end

(c::Conv)(X) = conv4(c.W, X, padding = c.padding, stride = c.stride,
                     dilation = c.dilation) .+ c.b

function reallocate(c::Conv, atype)
    Conv(Param(atype(value(c.W))), Param(atype(value(c.b))), c.stride,
         c.padding, c.dilation)
end

"""
    BatchNorm(dim, σ = identity)

Batch normalization.
"""
struct BatchNorm
    moments
    params
    σ
end
BatchNorm(dim, σ = identity) = BatchNorm(bnmoments(),
                                         DEFAULT_ARRAY_TYPE(bnparams(dim)), σ)
(bn::BatchNorm)(X) = bn.σ.(batchnorm(X, bn.moments, bn.params))

function reallocate(bn::BatchNorm, atype)
    mean, var = bn.moments.mean, bn.moments.var
    BatchNorm(
        bnmoments(;
            momentum = bn.moments.momentum,
            mean = isnothing(mean) ? nothing : atype(mean),
            var = isnothing(var) ? nothing : atype(var),
            meaninit = bn.moments.meaninit,
            varinit = bn.moments.varinit
        ),
        atype(value(bn.params)),
        bn.σ
    )
end

"""
    chain(layer1, layer2, ...)

Build a sequential model.
"""
struct Chain
    layers
    Chain(layers...) = new(layers)
end

function (c::Chain)(X)
    for layer in c.layers
        X = layer(X)
    end
    X
end

reallocate(chain::Chain, atype) = Chain((reallocate.(chain.layers, atype))...)

