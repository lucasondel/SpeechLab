# SPD-License-Identifier: MIT

using ArgParse
using CUDA
using Flux
using HDF5
using JLD2
using MarkovModels
using Random
using TOML
using Zygote

CUDA.allowscalar(false)

#======================================================================
Data loader to load batch of utterances.
======================================================================#
include("dataloader.jl")

#======================================================================
Differentiable LF-MMI objective function.
======================================================================#
include("lfmmi_loss.jl")

function train!(model, batchloader, denfsm, opt, θ, use_gpu)
    trainmode!(model)

    acc_loss::Float64 = 0
    N = 0

    for (i, (batch_data, batch_nums)) in enumerate(batchloader)
        batch_dens = union([denfsm for i in 1:size(batch_data, 3)]...)
        if use_gpu
            batch_data = cu(batch_data)
            batch_nums = MarkovModels.gpu(batch_nums)
        end

        L, gs = withgradient(θ) do
            lfmmi_loss(model(batch_data), batch_nums, batch_dens)
        end
        Flux.update!(opt, θ, gs)

        acc_loss += L
        N += 1
    end

    acc_loss / N
end

function test!(model, batchloader, denfsm, opt, θ, use_gpu)
    testmode!(model)

    acc_loss::Float64 = 0
    N = 0

    for (i, (batch_data, batch_nums)) in enumerate(batchloader)
        batch_dens = union([denfsm for i in 1:size(batch_data, 3)]...)
        if use_gpu
            batch_data = cu(batch_data)
            batch_nums = MarkovModels.gpu(batch_nums)
        end

        L, _ = lfmmi_loss(model(batch_data), batch_nums, batch_dens)

        acc_loss += L
        N += 1

    end

    acc_loss / N
end

function main(args)
    conf = TOML.parsefile(args["config"])
    lrate = conf["learning_rate"]
    epochs = conf["epochs"]
    mbsize = conf["mini_batch_size"]

    train_numfsms = load(args["train_numfsms"])
    dev_numfsms = load(args["dev_numfsms"])
    denfsm = load(args["denfsm"])["cfsm"]
    model = load(args["model"])["model"]
    use_gpu = args["use-gpu"]

    if args["use-gpu"]
        model = fmap(cu, model)
        denfsm = denfsm |> MarkovModels.gpu
    end

    θ = Flux.params(model)
    opt = ADAM(lrate)

    for epoch in 1:epochs
        train_loss = 0
        dev_loss = 0

        h5open(args["train"], "r") do trainfea
            trainbl = BatchLoader(trainfea, train_numfsms, mbsize)
            train_loss = train!(model, trainbl, denfsm, opt, θ, use_gpu)
        end

        h5open(args["dev"], "r") do devfea
            devbl = BatchLoader(devfea, dev_numfsms, mbsize)
            dev_loss = test!(model, devbl, denfsm, opt, θ, use_gpu)
        end

        println("epoch=$epoch/$epochs " *
                "train_loss=$(round(train_loss, digits = 4)) " *
                "dev_loss=$(round(dev_loss, digits = 4))")
    end
end


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--use-gpu", "-g"
            action = :store_true
            help = "train the model on gpu"
        "config"
            required = true
            help = "TOML configuration file for the training"
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
        "outdir"
            required = true
            help = "directory"
    end
    s.description = """
    Train a neural network using the LF-MMI objective function.
    of utterances to use is read from the standard input. Upon
    completion, the final model will be stored in outdir/final.bson.
    """
    parse_args(s)
end


args = parse_commandline()
main(args)
