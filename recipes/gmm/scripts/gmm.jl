# SPD-License-Identifier: MIT

using ArgParse
using BayesianModels
using CUDA
using BSON
using JLD2
using LinearAlgebra
using Random
using TOML

CUDA.allowscalar(false)

const DEFAULTS_MODEL = Dict(
    "ncomponents" => 1,
    "covtype" => "full",
    "priortype" => "categorical",
    "noise_init" => 0.1,
    "pstrength" => 1
)

const DEFAULTS_INFERENCE = Dict(
    "steps" => 1000,
    "learning_rate" => "0.1",
)

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--fea-ext", "-e"
            default = ".bson"
            help = "feature files extension"
        "--single-precision", "-s"
            action = :store_true
            help = "use float32 data type for computation"
        "--use-gpu", "-g"
            action = :store_true
            help = "use a GPU for training the model"
        "config"
            required = true
            help = "GMM configuration file"
        "stats"
            required = true
            help = "statistics of the data to initialize the model"
        "uttids"
            required = true
            help = "list of utterances to use for training"
        "feadir"
            required = true
            help = "directory containing the features"
        "gmm"
            required = true
            help = "trained GMM"
    end
    s.description = """
    Train a Gaussian Mixture Model.
    """
    parse_args(s)
end

function main(args)
    config = TOML.parsefile(args["config"])
    stats = BSON.load(args["stats"])
    modelconf = merge(DEFAULTS_MODEL, config["model"])
    inferenceconf = merge(DEFAULTS_INFERENCE, config["inference"])
    T = args["single-precision"] ? Float32 : Float64

    feafiles = []
    open(args["uttids"], "r") do f
        for line in eachline(f)
            push!(feafiles, line * args["fea-ext"])
        end
    end

    C = modelconf["ncomponents"]
    pstrength = modelconf["pstrength"]
    σ = modelconf["noise_init"]
    D = stats[:D]
    μ₀ = Array{T}(stats[:μ])
    W₀ = Array{T}(Hermitian(inv(stats[:Σ])))
    diagΣ₀ = Array{T}(diag(stats[:Σ]))

    @info "creating model with $C components (covtype = $(modelconf["covtype"])"
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
    model = Mixture(T, components = Tuple(components))

    if args["use-gpu"]
        init_gpu()
        model |> gpu!
    end

    @info "starting training..."
    lrate = inferenceconf["learning_rate"]
    steps = inferenceconf["steps"]
    mb_size = inferenceconf["mini_batch_size"]
    feadir = args["feadir"]
    params = filter(isbayesianparam, getparams(model))
    cache = Dict()
    for step in 1:steps
        # Clear cache for gradient computation
        empty!(cache)

        # Randomize the utterance list
        shuffle!(feafiles)[1:mb_size]

        # Load the minibatch data
        X = hcat([BSON.load(joinpath(feadir, feafiles[n]))[:data]
                  for n in 1:mb_size]...)
        X = convert(Array{T}, X)
        if args["use-gpu"] X = X |> CuArray end

        # One gradient step
        scale = length(feafiles) / mb_size
        𝓛 = elbo(model, X; cache = cache, stats_scale = scale)
        ∇ξ = ∇elbo(model, cache, params)
        gradstep(∇ξ, lrate = lrate)

        # Log the ELBO
        norm = scale * size(X,2)
        @info "step = $(step)/$(steps) 𝓛 = $(round((1/norm) * 𝓛, digits = 3))"
    end

    # Make sure the model is on CPU before saving it.
    model |> cpu!

    # Save the model and the config files.
    save(args["gmm"], Dict("modelconf" => modelconf,
                           "inferenceconf" => inferenceconf,
                           "model" => model))
end

args = parse_commandline()
main(args)
