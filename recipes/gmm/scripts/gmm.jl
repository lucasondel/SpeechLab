# SPD-License-Identifier: MIT

using ArgParse
using BayesianModels
using CUDA
using HDF5
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
        "--checkpoint-rate", "-c"
            default = 100
            arg_type = Int
            help = "save a checkpoint of the model every C updates"
        "--single-precision", "-S"
            action = :store_true
            help = "use float32 data type for computation"
        "--start-from", "-s"
            default = ""
            help = "start training from the given checkpoint model"
        "--start-update", "-k"
            default = 1
            arg_type = Int
            help = "start training the given update number"
        "--use-gpu", "-g"
            action = :store_true
            help = "use a GPU for training the model"
        "config"
            required = true
            help = "GMM configuration file"
        "stats"
            required = true
            help = "statistics of the data to initialize the model"
        "features"
            required = true
            help = "HDF5 archive containing the features"
        "outdir"
            required = true
            help = "directory"
    end
    s.description = """
    Train a Gaussian Mixture Model on a set of utterances. The list
    of utterances to use is read from the standard input. Upon
    completion, The final model will be stored in outdir/final.jld2.
    """
    parse_args(s)
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

function load_filelist(uttids, feadir, ext)
    filelist= []
    open(uttids, "r") do f
        for line in eachline(f)
            push!(filelist, joinpath(feadir, line*ext))
        end
    end
    filelist
end

function savemodel(dir, model, modelconf, inferenceconf, step = -1)
    data = Dict(
        "modelconf" => modelconf,
        "inferenceconf" => inferenceconf,
        "model" => model
    )
    modelname = step > 0 ? "$step.jld2" : "final.jld2"
    save(joinpath(dir, modelname), data)
end

function main(args)
    config = TOML.parsefile(args["config"])
    outdir = args["outdir"]
    T = args["single-precision"] ? Float32 : Float64

    utts = collect(eachline(stdin))

    model = nothing
    modelconf = nothing
    if isempty(args["start-from"])
        modelconf = merge(DEFAULTS_MODEL, config["model"])
        f = h5open(args["stats"], "r")
        μ = read(f["mean"])
        Σ = read(f["cov"])
        close(f)
        model = makemodel(T, modelconf, μ, Σ)
    else
        dict = load(args["start-from"])
        model = dict["model"]
        modelconf = dict["model"]
    end

    if args["use-gpu"]
        init_gpu()
        model |> gpu!
    end

    inferenceconf = merge(DEFAULTS_INFERENCE, config["inference"])
    checkrate = args["checkpoint-rate"]
    lrate = inferenceconf["learning_rate"]
    steps = inferenceconf["steps"]
    mb_size = inferenceconf["mini_batch_size"]
    params = filter(isbayesianparam, getparams(model))
    cache = Dict()
    h5open(args["features"], "r") do ffea
        for step in args["start-update"]:steps
            # Clear cache for gradient computation
            empty!(cache)

            # Randomize the utterance list
            shuffle!(utts)[1:mb_size]

            # Load the minibatch data
            X = hcat([read(ffea[utts[n]]) for n in 1:mb_size]...)
            X = convert(Array{T}, X)
            if args["use-gpu"] X = X |> CuArray end

            # One gradient step
            scale = length(utts) / mb_size
            𝓛, llh, KL = elbo(model, X; cache = cache, stats_scale = scale,
                              detailed = true)
            ∇ξ = ∇elbo(model, cache, params)
            gradstep(∇ξ, lrate = lrate)

            # Log the status of the training in JSON format
            norm = scale * size(X,2)
            norm𝓛 = 𝓛 / norm
            normllh = llh / norm
            normKL = KL / norm
            println("{\"step\": $step, \"nsteps\":$steps, \"elbo\": $norm𝓛, \"llh\":$normllh, \"kl\":$normKL}")

            # Checkpoint
            if step % checkrate == 0
                model |> cpu!
                savemodel(outdir, model, modelconf, inferenceconf, step)
                if args["use-gpu"] model |> gpu!  end
                params = filter(isbayesianparam, getparams(model))
            end
        end
    end

    model |> cpu!
    savemodel(outdir, model, modelconf, inferenceconf)
end

args = parse_commandline()
main(args)
