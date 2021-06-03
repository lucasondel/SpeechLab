# SPDX-License-Identifier: MIT

using ArgParse
using BSON
using ClusterManagers
using Distributed
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
        "--parallel-njobs", "-n"
            default = 1
            arg_type = Int
            help = "number of parallel jobs to use"
        "--parallel-args", "-a"
            default = ""
            arg_type = String
            help = "arguments to the parallel environment"
        "configfile"
            help = "features configuration file (TOML format)"
        "scp"
            help = "SCP file to extract the features"
        "outdir"
            help = "output directory where to store the feature files"
    end
    parse_args(s)
end

function main(args)
    conf = merge(DEFAULTS, TOML.parsefile(args["configfile"]))
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

    @info "extracting features to $(args["outdir"])..."

    @everywhere args = $args
    @everywhere conf = $conf
    @everywhere feaextractor = $feaextractor

    @sync @distributed for line in readlines(args["scp"])
        tokens = split(strip(line))
        uttid = tokens[1]
        path_or_pipe = join(tokens[2:end], " ")
        channels, srate = load(path_or_pipe)

        # Check that we can process the input
        srate == conf["srate"] || error("invalid sampling rate, expected $(conf["srate"]) got $srate")
        size(channels, 2) == 1 || error("cannot process more than 1 channel")

        fea = channels[:,1] |> feaextractor
        outpath = joinpath(args["outdir"], "$(uttid).bson")
        bson(outpath, Dict(:config => conf, :data => fea))
    end
end

args = parse_commandline()

@info "starting jobs..."
# Convert the string arguments to a Cmd object.
pargs = Cmd([String(token) for token in split(args["parallel-args"])])
addprocs_sge(args["parallel-njobs"], qsub_flags = pargs, exename = split("stdbuf -oL $(Sys.BINDIR)/julia"))
@info "started $(nworkers())/$(args["parallel-njobs"]) jobs"

@everywhere using BSON
@everywhere using SpeechFeatures
@everywhere using WAV

@everywhere loadpath(path) = wavread(path, format = "double")

@everywhere loadpipe(pipe) = channels, srate = wavread(IOBuffer(read(cmd)))
@everywhere function loadpipe(pipe)
    cmd = pipeline([`$(split(subcmd))` for subcmd in split(pipe[1:end-1], "|")]...)
    wavread(IOBuffer(read(cmd)))
end
@everywhere load(path_or_pipe) = endswith(path_or_pipe, "|") ? loadpipe(path_or_pipe) : loadpath(path_or_pipe)

main(args)

