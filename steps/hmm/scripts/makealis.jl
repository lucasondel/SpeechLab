# SPD-License-Identifier: MIT

using ArgParse
using JLD2
using MarkovModels

function main(args)
    hmmsdata = load(args["hmms"])
    lexicon = load(args["lexiconfsm"])
    hmms = hmmsdata["units"]
    pdfid_mapping = hmmsdata["pdfid_mapping"]
    sil = args["sil-sym"]
    bprob = args["between-silprob"]
    eprob = args["edge-silprob"]

    SF = LogSemifield{Float32}
    jldopen(args["alifsms"], "w") do f
        open(args["ali"], "r") do rf

            for line in eachline(rf)
                tokens = split(line)
                uttid, sentence = tokens[1], tokens[2:end]

                fsm = VectorFSM{SF}()

                # Initial silence state.
                prev = addstate!(fsm, sil;
                                 initweight = SF(log(eprob)),
                                 finalweight = zero(SF))

                for (i, word) in enumerate(sentence)
                    if word ∉ keys(lexicon)
                        @info "no pronunciation found for $word"
                        word = "<UNK>"
                    end

                    initweight = i == 1 ? SF(log(1-eprob)) : zero(SF)
                    finalweight = i == length(sentence) ? SF(log(1-eprob)) : zero(SF)
                    s = addstate!(fsm, word; initweight, finalweight)
                    addarc!(fsm, prev, s)

                    if 1 < i
                        silstate = addstate!(fsm, sil)
                        addarc!(fsm, prev, silstate, SF(log(bprob)))
                        addarc!(fsm, silstate, s, SF(log(bprob)))
                    end
                    prev = s
                end

                # Final silence state.
                silstate = addstate!(fsm, sil;
                                     initweight = zero(SF),
                                     finalweight = SF(log(eprob)))
                addarc!(fsm, prev, silstate)

                # G: Grammar L: Lexicon H: HMM
                GL = HierarchicalFSM(fsm |> renormalize, lexicon)
                GLH = HierarchicalFSM(GL, hmms)
                f["$uttid"] = MatrixFSM(GLH, pdfid_mapping, t -> (t[2], t[3]))
            end
        end
    end
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--sil-sym", "-s"
            default = "<SIL>"
            help = "optional silence symbol"
        "--edge-silprob", "-e"
            arg_type = Float64
            default = 0.8
            help = "probability of optional silence at the edges of the utterance"
        "--between-silprob", "-b"
            arg_type = Float64
            default = 0.2
            help = "probability of optional silence between words"
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

args = parse_commandline()
main(args)

