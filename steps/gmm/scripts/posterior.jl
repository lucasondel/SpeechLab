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
        "--compression-level", "-c"
            arg_type = Int
            default = 0
            help = "compression level of the HDF5 archive"
        "model"
            required = true
            help = "input GMM"
        "features"
            required = true
            help = "HDF5 archive containing the features"
        "out"
            required = true
            help = "output posteriors in HDF5 format"
    end
    s.description = """
    Compute the GMM's components posterior. The list of utterances
    to use is read from the standard input.
    """
    parse_args(s)
end

function main(args)
    model, _ = loadmodel(args["model"])

    h5open(args["out"], "w") do fout
        h5open(args["features"], "r") do ffea
            for utt in eachline(stdin)
                X = read(ffea[utt])
                fout[utt, compress = args["compression-level"]] = posterior(model, X)
            end
        end
    end
end

args = parse_commandline()
main(args)
