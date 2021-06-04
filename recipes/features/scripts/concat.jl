# SPDX-License-Identifier: MIT

using ArgParse
using HDF5

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--compression-level", "-c"
            arg_type = Int
            default = 0
            help = "compression level of the HDF5 archive"
        "out"
            help = "output HDF5 archive"
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
    h5open(args["out"], "w") do f

        # Use the first archive of the list to add the attributes to
        # the output archive.
        h5open(archives[1], "r") do farchive
            for k in keys(attributes(farchive))
                attributes(f)[k] = read(attributes(farchive)[k])
            end
        end

        # Copy the data of each archive in the new one.
        for arch in archives
            h5open(arch, "r") do farchive
                for k in keys(farchive["/"])
                    f[k, compress = args["compression-level"]] = read(farchive[k])
                end
            end
        end
    end
end

args = parse_commandline()
main(args)

