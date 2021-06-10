# SPD-License-Identifier: MIT

function load_filelist(uttids, feadir, ext)
    filelist= []
    open(uttids, "r") do f
        for line in eachline(f)
            push!(filelist, joinpath(feadir, line*ext))
        end
    end
    filelist
end

function makemodel(T, modelconf, μ, Σ)
    C = modelconf["ncomponents"]
    pstrength = modelconf["pstrength"]
    σ = modelconf["noise_init"]
    D = length(μ)
    μ₀ = Array{T}(μ)
    W₀ = Array{T}(Hermitian(inv(Σ)))
    diagΣ₀ = Array{T}(diag(Σ))

    @debug "creating model with $C components (covtype = $(modelconf["covtype"]))"
    components = []
    for c in 1:C
        if modelconf["covtype"] == "full"
            push!(components, Normal(T, D; μ₀, W₀, pstrength, σ))
        elseif modelconf["covtype"] == "diagonal"
            push!(components, NormalDiag(T, D; μ₀, diagΣ₀, pstrength, σ))
        else
            error("unknown cov type $(modelconf["covtype"])")
        end
    end
    Mixture(T, components = Tuple(components))
end

function savemodel(dir, model, modelconf, inferenceconf, step = -1)
    mtype = typeof(model)
    data = Dict(
        "modelconf" => modelconf,
        "inferenceconf" => inferenceconf,
        "mtype" => mtype,
        "mstate" => todict(model)
    )
    modelname = step > 0 ? "$step.jld2" : "final.jld2"

    save(joinpath(dir, modelname), data)
end

function loadmodel(path)
    dict = load(path)
    mtype = dict["mtype"]
    mstate = dict["mstate"]
    model = fromdict(mtype, mstate)
    modelconf = dict["modelconf"]
    model, modelconf
end
