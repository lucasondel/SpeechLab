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
    num_pdfs = hmmsdata["num_pdfs"]
    hmms = merge(su_hmms, nsu_hmms)

    SF = LogSemifield{Float32}
    fsm = VectorFSM{SF}()

    for unit in collect(keys(hmms))
        s = addstate!(fsm, unit)
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
    cfsm, labels = compile(fsm, num_pdfs)

    jldopen(args["cfsm"], "w") do f
        f["cfsm"] = cfsm
        f["labels"] = labels
    end
end

args = parse_commandline()
main(args)
