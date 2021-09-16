# SPD-License-Identifier: MIT

using ArgParse
using CUDA
using HDF5
using JLD2
using Knet
using ProgressMeter
using TOML

# Data loader to load batch of utterances.
include("dataloader.jl")

function main(args)
    use_gpu = args["use-gpu"]
    model = load(args["model"])["model"]
    getlengths_fn = s -> getlengths(TOML.parsefile(args["modelconfig"]), s)

    if args["use-gpu"]
        model = model |> gpu
    end

    testmode!(model)

    h5open(args["features"], "r") do fin
        h5open(args["out"], "w") do fout
            bl = DumpBatchLoader(fin, args["batch-size"])
            prog = Progress(length(bl))
            for (batch_data, inlengths, uttids) in bl
                if args["use-gpu"] batch_data = CuArray(batch_data) end

                seqlengths = getlengths_fn(inlengths)
                Y = Array(model(batch_data))
                for i in 1:size(batch_data, 3)
                    Yᵢ = Y[:, 1:seqlengths[i], i]
                    fout[uttids[i]] = Yᵢ
                end
                next!(prog)
            end
        end
    end
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--use-gpu", "-g"
            action = :store_true
            help = "train the model on gpu"
        "--batch-size", "-b"
            arg_type = Int
            default = 50
            help = "batch size"
        "modelfile"
            required = true
            help = "model definition file"
        "modelconfig"
            required = true
            help = "model configuration file in TOML format"
        "model"
            required = true
            help = "input model saved in JLD2 format"
        "features"
            required = true
            help = "input features in an HDF5 archive"
        "out"
            required = true
            help = "output HDF5 archive"
    end
    s.description = """
    Dump the output of the neural network.
    """
    parse_args(s)
end


args = parse_commandline()
include(args["modelfile"])
main(args)

