# SPD-License-Identifier: MIT

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
    uttids
end

function BatchLoader(h5data, alifsms, batchsize; shuffledata = false)
    uttids = if shuffledata
        shuffle(collect(keys(h5data)))
    else
        sort(keys(h5data), by = k -> size(h5data[k], 2))
    end
    BatchLoader(h5data, alifsms, batchsize, uttids)
end

Base.length(bl::BatchLoader) = Int(ceil(length(bl.h5data)/bl.batchsize))

function Base.iterate(bl::BatchLoader)
    iterate(bl, 1)
end

function Base.iterate(bl::BatchLoader, state)
    idx = state
    idx <= length(bl) || return nothing

    startidx = (idx-1) * bl.batchsize + 1
    endidx = min(startidx + bl.batchsize - 1, length(bl.h5data))

    utts = [read(bl.h5data[key]) for key in bl.uttids[startidx:endidx]]
    batch_alifsms = [bl.alifsms[key] for key in bl.uttids[startidx:endidx]]
    seqlengths = [size(utt, 2) for utt in utts]

    # Merge the fsms into one.
    batch_alis = union(batch_alifsms...)

    # Pad the utterances' features to get a 3D tensor.
    Nmax = maximum(seqlengths)
    batch_data = cat(padutterance.(utts, Nmax)..., dims = 3)

    (batch_data, batch_alis, seqlengths), idx+1
end

