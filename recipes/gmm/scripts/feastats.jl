# SPDX-License-Identifier: MIT

using ArgParse
using HDF5

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "features"
            required = true
            help = "HDF5 archive containing the features"
        "out"
            required = true
            help = "output HDF5 file containing the statistics"
    end
    s.description = """
    Extract statistics from the a set of utterances. The list
    of utterances to use is read from the standard input.
    """
    parse_args(s)
end

function main(args)
    h5open(args["features"], "r") do f

        # Initialize the accumulators with the 1st utterance
        data = read(f[readline(stdin)])
        D, N = size(data)
        x = dropdims(sum(data, dims = 2), dims = 2)
        xxᵀ = sum([c*c' for c in eachcol(data)])

        uttcount = 0
        for line in eachline(stdin)
            uttcount += 1
            @debug "reading $(uttcount)th utterance: $line"
            data = read(f[line])
            N += size(data, 2)
            x += dropdims(sum(data, dims = 2), dims = 2)
            xxᵀ += sum([c*c' for c in eachcol(data)])
        end
        μ = x ./ N
        Σ = (xxᵀ .- μ*μ') ./ N

        h5open(args["out"], "w") do fout
            fout["N"] = N
            fout["dim"] = D
            fout["mean"] = μ
            fout["cov"] = Σ
        end
    end
end

args = parse_commandline()
main(args)
