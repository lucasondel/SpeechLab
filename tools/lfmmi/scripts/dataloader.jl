# SPD-License-Identifier: MIT

using Random

function padutterance(utt, Nmax)
    D, N = size(utt)
    pad = similar(utt, D, Nmax - N)
    pad .= utt[:, end]
    hcat(utt, pad)
end

struct BatchLoader
    h5data
    alifsms
    batchsize
end

Base.length(bl::BatchLoader) = Int(ceil(length(bl.h5data)/bl.batchsize))

function Base.iterate(bl::BatchLoader)
    uttids = shuffle!(collect(keys(bl.h5data)))
    iterate(bl, (uttids, 1))
end

function Base.iterate(bl::BatchLoader, state)
    uttids, idx = state
    idx <= length(bl) || return nothing

    startidx = (idx-1) * bl.batchsize + 1
    endidx = min(startidx + bl.batchsize - 1, length(bl.h5data))

    utts = [read(bl.h5data[key]) for key in uttids[startidx:endidx]]
    batch_alis = union([alifsms[key] for key in uttids[staridx:endidx]]...)

    Nmax = maximum(size.(utts, 2))
    batch_data = cat(padutterance.(utts, Nmax)..., dims = 3)

    (batch_data, batch_alis), (uttids, idx+1)
end
