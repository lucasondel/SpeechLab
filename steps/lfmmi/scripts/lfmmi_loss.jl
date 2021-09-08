# SPD-License-Identifier: MIT

function lfmmi_loss(ϕ, numerator_fsms, denominator_fsms, seqlengths)
    GC.gc()
    ϕ = clamp.(ϕ, -30, 30)
    γ_num, ttl_num = pdfposteriors(numerator_fsms, ϕ, seqlengths)
    γ_den, ttl_den = pdfposteriors(denominator_fsms, ϕ, seqlengths)
    K, N, B = size(ϕ)
    loss = -sum((ttl_num .- ttl_den))
    grad = -(γ_num .- γ_den)
    loss, grad
end

Zygote.@adjoint function lfmmi_loss(ϕ, numerator_fsms, denominator_fsms,
                                    seqlengths)
    loss, grad = lfmmi_loss(ϕ, numerator_fsms, denominator_fsms, seqlengths)
    loss, Δ -> (Δ .* grad, nothing, nothing, nothing)
end
