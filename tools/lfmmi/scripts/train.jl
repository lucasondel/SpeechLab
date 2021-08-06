# SPD-License-Identifier: MIT

using ArgParse
using BSON
using CUDA
using Flux
using HDF5
using MarkovModels
using Zygote

CUDA.allowscalar(false)

include("dataloader.jl")

function lfmmi_loss(ϕ, numerator_fsms, denominator_fsm)
    γ_num, ttl_num = pdfposteriors(numerator_fsms, ϕ)
    batch_den = union([denominator_fsm for i in 1:size(ϕ, 3)]...)
    γ_den, ttl_den = pdfposteriors(batch_den, ϕ)
    loss = -sum((ttl_num .- ttl_den))
    grad = -(γ_num .- γ_den)
    loss, grad
end

Zygote.@adjoint function lfmmi_loss(ϕ, numerator_fsms, denominator_fsm)
    loss, grad = lfmmi_loss(ϕ, numerator_fsms, denominator_fsm)
    loss, Δ -> (Δ .* grad, nothing, nothing)
end
Zygote.refresh()

function main(args)
    trainfea = h5data(args["train"])
    numfsms = load(args["numfsms"])
    bl = BatchLoader(trainfea, alfsms, args["mini-batch-size"])
    denfsm = load(args["denfsm"])
    model = BSON.load(args["model"])[:model]

    opt = ADAM(args["learning-rate"])
    θ = params(model)
    for epoch in 1:args["epochs"]
        for (batch_data, batch_nums) in bl
            gs = gradients(θ) do
                L = lfmmi_loss(model(batch_data), batch_nums, denfsm)
                println("loss = $L")
            end
            update!(opt, θ, gs)
        end
    end
end


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--learning-rate", "-l"
            default = 0.1
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
            help = "training features"
        "numfsms"
            required = true
            help = "numerator fsms (i.e. alignment graphs)"
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
