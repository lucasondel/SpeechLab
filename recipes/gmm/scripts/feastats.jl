# SPDX-License-Identifier: MIT

using ArgParse
using BSON

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--fea-ext", "-e"
            default = ".bson"
            help = "feature files extension"
        "feadir"
            required = true
            help = "directory where are stored the feature files"
        "stats"
            required = true
            help = "output BSON file containing the statistics"
    end
    s.description = """
    Extract statistics from the a set of feature files. The list
    of utterances to use is read from the standard input.
    """
    parse_args(s)
end

function main(args)
    # Initialize the accumulators with the 1st utterance
    feadata = BSON.load(joinpath(args["feadir"], readline(stdin) * args["fea-ext"]))
    data = feadata[:data]
    D, N = size(data)
    x = dropdims(sum(data, dims = 2), dims = 2)
    xxᵀ = sum([c*c' for c in eachcol(data)])

    for line in eachline(stdin)
        feadata = BSON.load(joinpath(args["feadir"], line * args["fea-ext"]))
        data = feadata[:data]
        N += size(data, 2)
        x += dropdims(sum(data, dims = 2), dims = 2)
        xxᵀ += sum([c*c' for c in eachcol(data)])
    end
    μ = x ./ N
    Σ = (xxᵀ .- μ*μ') ./ N
    bson(args["stats"], Dict(:N => N, :μ => μ, :Σ => Σ, :D => D))
end

args = parse_commandline()
main(args)
