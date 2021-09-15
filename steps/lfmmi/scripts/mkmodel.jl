# SPD-License-Identifier: MIT

using ArgParse
using HDF5
using JLD2
using Knet
using MarkovModels
using TOML

function main(args)
    indim = h5open(args["trainfea"], "r") do f
        key = iterate(keys(f))[1]
        size(read(f, key), 1)
    end

    pdfid_mapping = load(args["hmms"])["pdfid_mapping"]
    outdim = maximum(values(pdfid_mapping))

    model = buildmodel(TOML.parsefile(args["config"]), indim, outdim)
    save(args["out"], Dict("model" => model))
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "modelfile"
            required = true
            help = "input model file"
        "config"
            required = true
            help = "TOML configuration file"
        "trainfea"
            required = true
            help = "training features"
        "hmms"
            required = true
            help = "hmms saved in JLD2 format"
        "out"
            required = true
            help = "output model"
    end
    s.description = """
    Create a randomly initialized model.
    """
    parse_args(s)
end

args = parse_commandline()
include(args["modelfile"])
main(args)

