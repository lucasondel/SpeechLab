# SPD-License-Identifier: MIT

using ArgParse
using JLD2
using MarkovModels
using TOML

function makehmm(unit, topo)
    SF = LogSemifield{Float32}
    fsm = VectorFSM{SF}()

    statecount = 0
    states = Dict()
    for state in topo["states"]
        initweight = SF(log(state["initweight"]))
        finalweight = SF(log(state["finalweight"]))
        s = addstate!(fsm, "$(state["id"])"; initweight, finalweight)
        states[state["id"]] = s
        statecount += 1
    end

    for arc in topo["arcs"]
        addarc!(fsm, states[arc["src"]], states[arc["dest"]],
                SF(log(arc["weight"])))
    end

    fsm |> renormalize, statecount
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nsu-topology", "-t"
            default = ""
            help = "use the given topology for the non speech units"
        "topology"
            required = true
            help = "hmm topology in TOML format"
        "units"
            required = true
            help = "list of units with their category (speech-units | nonspeech-units)"
        "hmms"
            required = true
            help = "output hmms in BSON format"
    end
    s.description = """
    Build a set of HMM. The topology file should be formatted as:


    states = [


    \ua0\ua0{ id = 1, initweight = 1.0, finalweight = 0.0 },


    \ua0\ua0...


    ]




    links = [


    \ua0\ua0{ src = 1, dest = 2, weight = 0.5 },


    \ua0\ua0...


    ]




    tiestates = false | true



    """
    parse_args(s)
end

function load(file)
    speech_units = Set()
    non_speech_units = Set()
    open(file, "r") do f
        for line in eachline(f)
            tokens = split(line)
            if tokens[2] == "speech-unit"
                push!(speech_units, tokens[1])
            elseif tokens[2] == "nonspeech-unit"
                push!(non_speech_units, tokens[1])
            else
                throw(ArgumentError("uknown unit type: $(tokens[2])"))
            end
        end
    end
    speech_units, non_speech_units
end

function main(args)
    topo = TOML.parsefile(args["topology"])
    nsu_topo = if args["nsu-topology"] == ""
        topo
    else
        TOML.parsefile(args["nsu-topology"])
    end

    speech_units, non_speech_units = load(args["units"])

    num_pdfs = 0
    su_hmms = Dict()
    for unit in speech_units
        hmm, statecount = makehmm(unit, topo)
        su_hmms[unit] = hmm
        num_pdfs += statecount
    end

    nsu_hmms = Dict()
    for unit in non_speech_units
        hmm, statecount = makehmm(unit, nsu_topo)
        nsu_hmms[unit] = hmm
        num_pdfs += statecount
    end

    save(args["hmms"], Dict("speech_units" => su_hmms,
                            "non_speech_units" => nsu_hmms,
                            "num_pdfs" => num_pdfs))
end

args = parse_commandline()
main(args)

