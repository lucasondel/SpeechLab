# SPD-License-Identifier: MIT

using ArgParse
using JLD2
using MarkovModels

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "lexicon"
            required = true
            help = "input lexicon file"
        "lexiconfsm"
            required = true
            help = "output lexicon in JLD2 format"
    end
    s.description = """
    Build the lexicon FSM. The lexicon input file should be formatted
    as:

    word1 p r o n u n

    word1 p R o N u n

    word2 w o r d
    """
    parse_args(s)
end

function LinearFSM(seq)
    SF = LogSemifield{Float32}
    fsm = VectorFSM{SF}()

    prev = nothing
    for (i, label) in enumerate(seq)
        initweight = i == 1 ? one(SF) : zero(SF)
        finalweight = i == length(seq) ? one(SF) : zero(SF)
        s = addstate!(fsm, label; initweight, finalweight)
        prev = s
    end

    fsm
end

function main(args)
    pronuns = Dict()
    open(args["lexicon"], "r") do f
        for line in eachline(f)
            tokens = split(line)
            word = tokens[1]
            pronun = tokens[2:end]
            list = get(pronuns, word, [])
            push!(list, pronun)
            pronuns[word] = list
        end
    end

    jldopen(args["lexiconfsm"], "w") do f
        for word in keys(pronuns)
            fsms = [LinearFSM(pronun) for pronun in pronuns[word]]
            fsm = union(fsms...)
            f[word] = fsm |> minimize |> renormalize

        end
    end
end

args = parse_commandline()
main(args)
