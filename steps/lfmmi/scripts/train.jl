# SPD-License-Identifier: MIT

using ArgParse
using CUDA
using Dates
using Flux
using HDF5
using JLD2
using MarkovModels
using Random
using TOML
using Zygote

CUDA.allowscalar(false)

# Data loader to load batch of utterances.
include("dataloader.jl")

# Differentiable LF-MMI objective function.
include("lfmmi_loss.jl")

function train!(model, batchloader, denfsm, opt, θ, use_gpu, getlengths)
    trainmode!(model)

    acc_loss::Float64 = 0
    N = 0

    for (i, (batch_data, batch_nums, inlengths)) in enumerate(batchloader)
        seqlengths = getlengths(inlengths)

        batch_dens = union([denfsm for i in 1:size(batch_data, 3)]...)
        if use_gpu
            batch_data = cu(batch_data)
            batch_nums = MarkovModels.gpu(batch_nums)
        end

        @debug "seqlengths $seqlengths"
        @debug "$(size(batch_data))"
        @debug "typeof(batch_data) = $(typeof(batch_data))"

        L, gs = withgradient(θ) do
            lfmmi_loss(model(batch_data), batch_nums, batch_dens, seqlengths)
        end
        Flux.update!(opt, θ, gs)

        acc_loss += L / sum(seqlengths)
        N += 1
    end

    acc_loss / N
end

function test!(model, batchloader, denfsm, opt, θ, use_gpu, getlengths)
    testmode!(model)

    acc_loss::Float64 = 0
    N = 0

    for (i, (batch_data, batch_nums, inlengths)) in enumerate(batchloader)
        seqlengths = getlengths(inlengths)

        batch_dens = union([denfsm for i in 1:size(batch_data, 3)]...)

        if use_gpu
            batch_data = CuArray(batch_data)
            batch_nums = MarkovModels.gpu(batch_nums)
        end

        L, _ = lfmmi_loss(model(batch_data), batch_nums, batch_dens, seqlengths)

        acc_loss += L
        N += sum(seqlengths)
    end

    acc_loss / N
end

function main(args)
    conf = TOML.parsefile(args["config"])
    epochs = conf["epochs"]
    mbsize = conf["mini_batch_size"]
    use_gpu = args["use-gpu"]
    cdir = args["checkpoint-dir"]
    ckptpath = args["from-checkpoint"]

    ckpt = if ckptpath != ""
        @debug "Using checkpoint: $ckptpath."
        load(ckptpath)
    else
        nothing
    end

    getlengths_fn = s -> getlengths(TOML.parsefile(args["modelconfig"]), s)

    train_numfsms = load(args["train_numfsms"])
    dev_numfsms = load(args["dev_numfsms"])
    denfsm = load(args["denfsm"])["fsm"]

    model = isnothing(ckpt) ? load(args["model"])["model"] : ckpt["model"]
    opt = isnothing(ckpt) ? getoptimizer(conf) : ckpt["optimizer"]
    scheduler = isnothing(ckpt) ? getscheduler(conf) : ckpt["scheduler"]
    best_loss = isnothing(ckpt) ? Float32(Inf) : ckpt["best_loss"]
    startepoch = isnothing(ckpt) ? 1 : ckpt["epoch"]+1

    @debug "model = $model"

    if args["use-gpu"]
        model = fmap(cu, model)
        denfsm = denfsm |> MarkovModels.gpu
    end

    θ = Flux.params(model)
    for epoch in startepoch:epochs
        t₀ = now()

        train_loss = 0
        dev_loss = 0
        shuffledata = epoch > conf["curriculum"] ? true : false
        if ! shuffledata
            @debug "Loading utterances sorted by length (i.e. curriculum training)."
        end

        h5open(args["train"], "r") do trainfea
            trainbl = BatchLoader(trainfea, train_numfsms, mbsize; shuffledata)
            train_loss = train!(model, trainbl, denfsm, opt, θ, use_gpu,
                                getlengths_fn)
        end

        h5open(args["dev"], "r") do devfea
            devbl = BatchLoader(devfea, dev_numfsms, mbsize)
            dev_loss = test!(model, devbl, denfsm, opt, θ, use_gpu,
                             getlengths_fn)
        end

        update!(scheduler, opt, dev_loss)

        checkpoint = Dict(
            "model" => model |> cpu,
            "optimizer" => opt,
            "epoch" => epoch,
            "best_loss" => best_loss,
            "scheduler" => scheduler
        )
        ckpt_path = joinpath(cdir, "last.jld2")
        @debug "Saving checkpoint to $ckpt_path."
        save(ckpt_path, checkpoint)

        if dev_loss < best_loss
            best_loss = dev_loss
            ckpt_path = joinpath(cdir, "best.jld2")
            @debug "New best lost is $best_loss; model saved to $ckpt_path."
            save(ckpt_path, checkpoint)
        end

        t₁ = now()
        println("epoch=$epoch/$epochs " *
                "train_loss=$(round(train_loss, digits = 4)) " *
                "dev_loss=$(round(dev_loss, digits = 4)) " *
                "epoch_duration=$((t₁ - t₀).value/1000)")
    end

    @debug "Saving final model to $(args["out"])."
    save(args["out"], Dict("model" => model |> cpu))
end


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--checkpoint-dir", "-c"
            default = "checkpoints"
            help = "directory where to save training checkpoint"
        "--from-checkpoint", "-f"
            default = ""
            help = "start training from the provided checkpoint"
        "--use-gpu", "-g"
            action = :store_true
            help = "train the model on gpu"
        "modelfile"
            required = true
            help = "model definition file"
        "config"
            required = true
            help = "training configuration file (TOML)"
        "modelconfig"
            required = true
            help = "model configuration file (TOML)"
        "train"
            required = true
            help = "training set"
        "dev"
            required = true
            help = "development set"
        "train_numfsms"
            required = true
            help = "numerator fsms (i.e. alignment graphs) for the training set"
        "dev_numfsms"
            required = true
            help = "numerator fsms (i.e. alignment graphs) for the dev set"
        "denfsm"
            required = true
            help = "denominator graph"
        "model"
            required = true
            help = "input Flux model saved in BSON format"
        "out"
            required = true
            help = "output model"
    end
    s.description = """
    Train a neural network using the LF-MMI objective function.
    of utterances to use is read from the standard input.
    """
    parse_args(s)
end


args = parse_commandline()

# This file should define:
#   * `getlengths`
#   * `getoptimizer`
#   * `getscheduler`
include(args["modelfile"])
main(args)

