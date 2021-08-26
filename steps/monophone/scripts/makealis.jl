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
                uttid, sentence = tokens[1], tokens[2:end]

                fsm = VectorFSM{SF}()
                prev = nothing
                for (i, word) in enumerate(sentence)
                    if word ∉ keys(lexicon)
                        @info "no pronunciation found for $word"
                        word = "<UNK>"
                    end

                    initweight = i == 1 ? one(SF) : zero(SF)
                    finalweight = i == length(sentence) ? one(SF) : zero(SF)
                    s = addstate!(fsm, word; initweight, finalweight)
                    if i > 1
                        addarc!(fsm, prev, s)
                    end
                    prev = s
                end

                # G: Grammar L: Lexicon H: HMM
                GL = HierarchicalFSM(fsm |> renormalize, lexicon)
                GLH = HierarchicalFSM(GL, hmms)
                println("MatrixFSM()")
                #cfsm = MatrixFSM()
                #f["$uttid/cfsm"] = cfsm
                #f["$uttid/labels"] = labels
            end
        end
    end
end

args = parse_commandline()
main(args)
