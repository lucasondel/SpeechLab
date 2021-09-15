# SPD-License-Identifier: MIT

using AutoGrad

function _lfmmi_loss(ϕ, numfsm, denfsm, seqlengths)
    GC.gc()
    CUDA.reclaim()
    ϕ = clamp.(ϕ, -30, 30)
    γ_num, ttl_num = pdfposteriors(numfsm, ϕ, seqlengths)
    γ_den, ttl_den = pdfposteriors(denfsm, ϕ, seqlengths)
    K, N, B = size(ϕ)
    loss = -sum((ttl_num .- ttl_den))
    loss, γ_num, γ_den
end

function _∇lfmmi_loss(dy, loss, γ_num, γ_den)
    retval = -(γ_num .- γ_den)
end

@primitive1 _lfmmi_loss(ϕ, numf, denf, slen),dy,y (dy[1] * _∇lfmmi_loss(dy, y...))

lfmmi_loss(ϕ, numfsm, denfsm, seqlengths) =
    _lfmmi_loss(ϕ, numfsm, denfsm, seqlengths)[1] # return the loss only
