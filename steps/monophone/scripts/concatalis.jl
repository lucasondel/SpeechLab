# SPDX-License-Identifier: MIT

using ArgParse
using JLD2
using MarkovModels

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "out"
            help = "output JLD2 archive"
            required = true
        "archives"
            nargs = '+'
            help = "archives to concatenate"
            required = true
    end
    parse_args(s)
end

function main(args)
    archives = args["archives"]
    jldopen(args["out"], "w") do f

        # Copy the data of each archive in the new one.
        for arch in archives
            jldopen(arch, "r") do farchive
                for uttid in keys(farchive)
                    f["$uttid"] = farchive["$uttid"]
                end
            end
        end
    end

end

args = parse_commandline()
main(args)

