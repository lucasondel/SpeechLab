# SPD-License-Identifier: MIT

using ArgParse
using JLD2
using MarkovModels

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--start-non-speech", "-s"
            action = :store_true
            help = "force the graph to start with a non-speech unit"
        "--end-non-speech", "-e"
            action = :store_true
            help = "force the graph to end with a non-speech unit"
        "hmms"
            required = true
            help = "per-unit hmms"
        "cfsm"
            required = true
            help = "output compiled fsm"
    end
    s.description = """
    Build a phone-loop decoding graph.
    """
    parse_args(s)
end

function main(args)
    hmmsdata = load(args["hmms"])
    su_hmms = hmmsdata["speech_units"]
    nsu_hmms = hmmsdata["non_speech_units"]
    hmms = merge(su_hmms, nsu_hmms)

    SF = LogSemifield{Float64}
    fsm = FSM{SF}()

    for unit in collect(keys(hmms))
        s = addstate!(fsm, label = unit)
        if ! args["start-non-speech"]
            setinit!(s)
        elseif unit ∈ keys(nsu_hmms)
            setinit!(s)
        end

        if ! args["end-non-speech"]
            setfinal!(s)
        elseif unit ∈ keys(nsu_hmms)
            setfinal!(s)
        end
    end

    for src in states(fsm)
        for dest in states(fsm)
            link!(fsm, src, dest)
        end
    end
    fsm = replace(renormalize!(fsm), hmms)
    cfsm = compile(fsm)

    jldopen(args["cfsm"], "w") do f f["cfsm"] = cfsm end
end

args = parse_commandline()
main(args)
