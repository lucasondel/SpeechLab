# SPDX-License-Identifier: MIT

using ArgParse
using HDF5
using SpeechFeatures
using TOML
using WAV

#######################################################################
# Default values

const FEATYPES = Dict(
    "MFCC" => (MFCC,
               ("removedc", "dithering", "srate", "frameduration",
                "framestep", "preemphasis", "windowfn", "windowpower",
                "nfilters", "lofreq", "hifreq", "addenergy", "nceps",
                "liftering")
              ),
    "FBANK" => (LogMelSpectrum,
                ("removedc", "dithering", "srate", "frameduration",
                 "framestep", "preemphasis", "windowfn", "windowpower",
                 "nfilters", "lofreq", "hifreq")
               ),
    "MAGSPEC" => (LogMagnitudeSpectrum,
                  ("removedc", "dithering", "srate", "frameduration",
                   "framestep", "preemphasis", "windowfn", "windowpower")
                 )
)

const WINDOWS = Dict(
    "hann" => SpeechFeatures.HannWindow,
    "hamming" => SpeechFeatures.HammingWindow,
    "rectangular" => SpeechFeatures.RectangularWindow
)

const DEFAULTS = Dict(
    "removedc" => true,
    "dithering" => 0,
    "srate" => 16000,
    "frameduration" => 0.025,
    "framestep" => 0.01,
    "preemphasis" => 0.97,
    "windowfn" => "hann",
    "windowpower" => 0.85,
    "nfilters" => 26,
    "lofreq" => 80,
    "hifreq" => 7600,
    "addenergy" => true,
    "nceps" => 12,
    "liftering" => 22
)

const DEFAULTS_DELTA = Dict(
    "order" => 2,
    "deltawin" => 2
)

const DEFAULTS_NORM = Dict(
    "mean_norm" => true,
)

#######################################################################

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--compression-level", "-c"
            arg_type = Int
            default = 0
            help = "compression level of the HDF5 archive"
        "configfile"
            required = true
            help = "features configuration file (TOML format)"
        "scp"
            required = true
            help = "SCP file to extract the features"
        "out"
            required = true
            help = "output HDF5 archive"
    end
    parse_args(s)
end

loadpath(path) = wavread(path, format = "double")

loadpipe(pipe) = channels, srate = wavread(IOBuffer(read(cmd)))
function loadpipe(pipe)
    subcmds = [`$(split(subcmd))` for subcmd in split(pipe[1:end-1], "|")]
    cmd = pipeline(subcmds...)
    wavread(IOBuffer(read(cmd)))
end

load(path_or_pipe) =
    endswith(path_or_pipe, "|") ? loadpipe(path_or_pipe) : loadpath(path_or_pipe)

function load_scpentry(line)
    tokens = split(strip(line))
    uttid = tokens[1]
    path_or_pipe = join(tokens[2:end], " ")
    String(uttid), path_or_pipe
end

function buildextractor(conf)
    featype, options = FEATYPES[conf["featype"]]
    feaopts = Dict()
    for opt in options
        if opt == "windowfn"
            feaopts[Symbol(opt)] = WINDOWS[conf[opt]]
        else
            feaopts[Symbol(opt)] = conf[opt]
        end
    end
    feaextractor = featype(;feaopts...)

    if "DeltaCoeffs" in keys(conf)
        deltaopts = merge(DEFAULTS_DELTA, conf["DeltaCoeffs"])
        Δ_extractor = DeltaCoeffs(order = deltaopts["order"],
                                  deltawin = deltaopts["deltawin"])
        feaextractor = Δ_extractor ∘ feaextractor
    end

    if "Normalization" in keys(conf)
        normopts = merge(DEFAULTS_NORM, conf["Normalization"])
        if normopts["mean_norm"]
            feaextractor = MeanNorm() ∘ feaextractor
        end
    end

    feaextractor
end

function main(args)
    conf = merge(DEFAULTS, TOML.parsefile(args["configfile"]))
    feaextractor = buildextractor(conf)

    @debug "extracting features to $(args["out"])"

    h5open(args["out"], "w") do f

        for (key, val) in conf
            if typeof(val) <: Dict
                for (skey, sval) in val
                    attributes(f)["$key.$skey"] = sval
                end
            else
                attributes(f)[key] = val
            end
        end

        for line in readlines(args["scp"])
            @debug "processing $line"
            uttid, path_or_pipe = load_scpentry(line)
            channels, srate = load(path_or_pipe)


            @assert srate == conf["srate"]
            @assert size(channels, 2) == 1

            fea = channels[:,1] |> feaextractor
            fea = convert(Array{Float32}, fea)

            f[uttid, compress = args["compression-level"]] = fea
        end
    end
end

run(`hostname`)
println(ARGS)
args = parse_commandline()
main(args)

