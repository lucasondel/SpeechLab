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

include("utils.jl")

const DEFAULTS_MODEL = Dict(
    "ncomponents" => 1,
    "covtype" => "full",
    "priortype" => "categorical",
    "noise_init" => 0.1,
    "pstrength" => 1
)

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--checkpoint-rate", "-c"
            default = 100
            arg_type = Int
            help = "save a checkpoint of the model every C updates"
        "--learning-rate", "-l"
            default = 0.1
            arg_type = Float64
            help = "constant learning rate of the training"
        "--mini-batch-size", "-m"
            default = 1
            arg_type = Int
            help = "number of utterances per mini-batch"
        "--single-precision", "-S"
            action = :store_true
            help = "use float32 data type for computations"
        "--start-from", "-s"
            default = ""
            help = "start training from the given checkpoint model"
        "--start-update", "-k"
            default = 1
            arg_type = Int
            help = "start training the given update number"
        "--steps", "-n"
            default = 1
            arg_type = Int
            help = "number of training steps"
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
        model, modelconf = loadmodel(args["start-from"])
    end

    if args["use-gpu"]
        init_gpu()
        model |> gpu!
    end

    checkrate = args["checkpoint-rate"]
    lrate = args["learning-rate"]
    steps = args["steps"]
    mb_size = args["mini-batch-size"]
    inferenceconf = Dict(
        "learning-rate" => lrate,
        "steps" => steps,
        "mb_size" => mb_size
    )
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
