# SPD-License-Identifier: MIT

using ArgParse
using JLD2
using MarkovModels

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "hmms"
            required = true
            help = "per-unit hmms"
        "lexiconfsm"
            required = true
            help = "input lexicon fsm"
        "ali"
            required = true
            help = "alignment file"
        "alifsms"
            required = true
            help = "output alignment fsms"
    end
    s.description = """
    Build alignment fsms.
    """
    parse_args(s)
end

function main(args)
    hmmsdata = load(args["hmms"])
    lexicon = load(args["lexiconfsm"])
    su_hmms = hmmsdata["speech_units"]
    nsu_hmms = hmmsdata["non_speech_units"]
    num_pdfs = hmmsdata["num_pdfs"]
    hmms = merge(su_hmms, nsu_hmms)

    SF = LogSemifield{Float32}

    jldopen(args["alifsms"], "w") do f
        open(args["ali"], "r") do rf

            for line in eachline(rf)
                tokens = split(line)
                uttid = tokens[1]

                fsm = FSM{SF}()

                prev = nothing
                for (i, token) in enumerate(tokens[2:end])
                    @assert token in keys(hmms) "unkown unit $token"

                    s = addstate!(fsm, label = token)
                    if i > 1
                        link!(fsm, prev, s)
                    else
                        setinit!(s)
                    end
                    prev = s
                end
                setfinal!(prev)

                fsm = replace(replace(renormalize!(fsm), lexicon), hmms)
                f[uttid] = compile(fsm |> remove_eps, num_pdfs)
            end
        end
    end
end

args = parse_commandline()
main(args)
