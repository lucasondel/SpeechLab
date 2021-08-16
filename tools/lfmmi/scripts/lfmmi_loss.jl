# SPD-License-Identifier: MIT

function lfmmi_loss(ϕ, numerator_fsms, denominator_fsms)
    γ_num, ttl_num = pdfposteriors(numerator_fsms, ϕ)
    γ_den, ttl_den = pdfposteriors(denominator_fsms, ϕ)
    K, N, B = size(ϕ)
    loss = -sum((ttl_num .- ttl_den)) / (N*B)
    grad = -(γ_num .- γ_den)
    loss, grad
end

Zygote.@adjoint function lfmmi_loss(ϕ, numerator_fsms, denominator_fsms)
    loss, grad = lfmmi_loss(ϕ, numerator_fsms, denominator_fsms)
    loss, Δ -> (Δ .* grad, nothing, nothing)
end
