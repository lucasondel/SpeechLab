# SPD-License-Identifier: MIT

using ArgParse
using BayesianModels
using CUDA
using HDF5
using JLD2
using LinearAlgebra
using Random

CUDA.allowscalar(false)

include("utils.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "model"
            required = true
            help = "input GMM"
        "features"
            required = true
            help = "HDF5 archive containing the features"
    end
    s.description = """
    Compute the GMM most likely components. The list of utterances
    to use is read from the standard input and the results is output
    on the standard output.
    """
    parse_args(s)
end

function main(args)
    model, _ = loadmodel(args["model"])
    h5open(args["features"], "r") do ffea
        for utt in eachline(stdin)
            X = read(ffea[utt])
            println("$utt\t$(join(predict(model, X), " "))")
        end
    end
end

args = parse_commandline()
main(args)
