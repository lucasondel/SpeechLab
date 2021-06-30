# SPD-License-Identifier: MIT
#
using ArgParse
using JLD2
using MarkovModels
using TOML

function makehmm(topo, pdfstartidx)
    SF = LogSemifield{Float64}
    fsm = FSM{SF}()

    states = Dict()
    newidx = pdfstartidx
    for state in topo["states"]
        if newidx == pdfstartidx || ! topo["tiestates"]
            newidx += 1
        end

        initweight = SF(log(state["initweight"]))
        finalweight = SF(log(state["finalweight"]))
        states[state["id"]] = addstate!(fsm, pdfindex = newidx,
                                        initweight = initweight,
                                        finalweight = finalweight,
                                        label = "$(state["id"])")
    end

    for link in topo["links"]
        weight = SF(log(link["weight"]))
        link!(fsm, states[link["src"]], states[link["dest"]], weight)
    end
    fsm, newidx
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
            help = "list of units with their category (speech-units | non-speech-units)"
        "hmms"
            required = true
            help = "output hmms in BSON format"
    end
    s.description = """
    Build a set of HMM. The topology file should be formatted as:




    \ua0\ua0states = [


    \ua0\ua0\ua0\ua0{ id = 1, initweight = 1.0, finalweight = 0.0 },


    \ua0\ua0\ua0\ua0...


    \ua0\ua0]




    \ua0\ua0links = [


    \ua0\ua0\ua0\ua0{ src = 1, dest = 2, weight = 0.5 },


    \ua0\ua0\ua0\ua0...


    \ua0\ua0]




    \ua0\ua0tiestates = false | true


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
            elseif tokens[2] == "non-speech-unit"
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
    nsu_topo = args["nsu-topology"] == "" ? topo : TOML.parsefile(args["nsu-topology"])

    speech_units, non_speech_units = load(args["units"])

    pdfstartidx = 0

    su_hmms = Dict()
    for unit in speech_units
        hmm, pdfstartidx = makehmm(topo, pdfstartidx)
        su_hmms[unit] = hmm
    end

    nsu_hmms = Dict()
    for unit in non_speech_units
        hmm, pdfstartidx = makehmm(nsu_topo, pdfstartidx)
        nsu_hmms[unit] = hmm
    end

    save(args["hmms"], Dict("speech_units" => su_hmms,
                            "non_speech_units" => nsu_hmms))
end

args = parse_commandline()
main(args)
