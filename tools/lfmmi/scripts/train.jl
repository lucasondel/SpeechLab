# SPD-License-Identifier: MIT

using ArgParse
using CUDA
using Flux
using HDF5
using JLD2
using JSON
using MarkovModels
using Random
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

        L, _ = lfmmi_loss(model(batch_data), batch_nums, batch_dens)
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

        L, gs = withgradient(θ) do
            lfmmi_loss(model(batch_data), batch_nums, batch_dens)
        end
        acc_loss += L
        N += 1

    end

    acc_loss / N
end

function main(args)
    train_numfsms = load(args["train_numfsms"])
    dev_numfsms = load(args["dev_numfsms"])
    denfsm = load(args["denfsm"])["cfsm"]
    model = load(args["model"])["model"]
    use_gpu = args["use-gpu"]
    mbsize = args["mini-batch-size"]

    if args["use-gpu"]
        model = fmap(cu, model)
        denfsm |> MarkovModels.gpu
    end

    θ = Flux.params(model)
    opt = ADAM(args["learning-rate"])

    h5open(args["train"], "r") do trainfea
        for epoch in 1:args["epochs"]
            trainloss::Float64 = 0
            devloss::Float64 = 0
            h5open(args["train"], "r") do trainfea
                trainbl = BatchLoader(trainfea, train_numfsms, mbsize)
                train_oss = train!(model, trainbl, denfsm, opt, θ, use_gpu)
            end
            h5open(args["dev"], "r") do devfea
                devbl = BatchLoader(devfea, dev_numfsms, mbsize)
                dev_oss = test!(model, devbl, denfsm, opt, θ, use_gpu)
            end

            println("epoch $epoch/$(args["epochs"]) " *
                    "training loss = $(round(train_loss, digits = 4)) " *
                    "dev loss = $(round(dev_loss, digits = 4))")
        end
    end
end


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--learning-rate", "-l"
            default = 1e-3
            arg_type = Float64
            help = "initial learning rate of the ADAM optimizer"
        "--mini-batch-size", "-m"
            default = 1
            arg_type = Int
            help = "number of utterances per mini-batch"
        "--epochs", "-e"
            default = 1
            arg_type = Int
            help = "number of training epochs"
        "--use-gpu", "-g"
            action = :store_true
            help = "use a GPU for training the model"
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
