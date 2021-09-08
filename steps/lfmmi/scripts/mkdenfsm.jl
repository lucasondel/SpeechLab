# SPD-License-Identifier: MIT

using ArgParse
using JLD2
using MarkovModels
using Random

function loadlexicon(path)
    retval = Dict()
    open(path, "r") do f
        for line in eachline(f)
            tokens = split(line)
            word, pronun = tokens[1], tokens[2:end]
            pronuns = push!(get(retval, word, []), pronun)
            retval[word] = pronuns
        end
    end
    retval
end

function rand_addsil(sentence, silsym, between_silprob, edge_silprob)
    retval = rand() < edge_silprob ? [silsym] : []
    for word in sentence[1:end-1]
        push!(retval, word)
        if rand() < between_silprob
            push!(retval, silsym)
        end
    end
    push!(retval, sentence[end])
    if rand() < edge_silprob
        push!(retval, silsym)
    end
    retval
end

function main(args)
    oovsym = args["oov-sym"]
    silsym = args["sil-sym"]
    between_silprob = args["between-silprob"]
    edge_silprob = args["edge-silprob"]
    lexicon = loadlexicon(args["lexicon"])
    n = args["ngram-order"]
    hmms = load(args["hmms"])

    ###################################################################
    # Estimate the n-gram count without pruning

    @info "Estimating $(n)-gram phonotactic language model."

    counts = Dict()
    for line in eachline(stdin)
        tokens = split(line)
        tokens = rand_addsil(tokens, silsym, between_silprob, edge_silprob)
        seq = []
        for token in tokens
            if token ∈ keys(lexicon)
                push!(seq, lexicon[token][1])
            else
                push!(seq, lexicon[oovsym][1])
            end
        end

        # Append nothing to mark the end of sentence.
        seq = vcat(seq..., [nothing])

        for i in 1:length(seq)
            src = tuple(seq[max(1, i-n+1):i-1]...)
            dest = if n > 1
                tuple(seq[max(1, i-n+2):i]...)
            else
                tuple(seq[i])
            end
            counts[(src, dest)] = get(counts, (src, dest), 0) + 1
        end
    end

    ###################################################################
    # Build an FSM from the counts

    @info "Representing the language model as a FSM."

    SF = LogSemifield{Float32}
    fsm = VectorFSM{SF}()

    # Get the init and final weights
    iws = Dict()
    fws = Dict()
    for ((src, dest), count) in counts
        if isempty(src) iws[dest] = SF(log(count)) end
        if isnothing(dest[end]) fws[src] = SF(log(count)) end
    end

    # Create the states.
    smap = Dict()
    for ((src, dest), count) in counts
        if ! isnothing(dest[end]) && dest ∉ keys(smap)
            initweight = get(iws, dest, zero(SF))
            finalweight = get(fws, dest, zero(SF))

            # This will be nonzero only in the unigram case.
            finalweight += get(fws, tuple(), zero(SF))

            smap[dest] = addstate!(fsm, dest; initweight, finalweight)
        end
    end

    # Create the arcs.
    for ((src, dest), count) in counts
        # Not an initial / final pseudo-arc.
        if ! isnothing(src) && ! isnothing(dest[end])
            if src == tuple() && n == 1
                # special case for unigram probabilty:
                # we use the same weight for each possible src states
                for s in values(smap)
                    addarc!(fsm, s, smap[dest], SF(log(count)))
                end
            elseif src ≠ tuple()
                addarc!(fsm, smap[src], smap[dest], SF(log(count)))
            end
        end
    end

    fsm = fsm

    ###################################################################
    # Compile and save

    @info "Composing the language model with units' HMM."

    # The previously built FSM combines the grammar and the lexicon.
    GL = fsm |> renormalize
    H = hmms["units"]
    GLH = HierarchicalFSM(GL, H)

    @info "Compiling."

    # Compile the expanded FSM into a matrix format.
    mfsm = MatrixFSM(GLH, hmms["pdfid_mapping"], t -> (t[end-1], t[end]))

    save(args["denfsm"], Dict("fsm" => mfsm))
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--oov-sym"
            default = "<UNK>"
            help = "fallback for out of vocabulary words"
        "--sil-sym"
            default = "<SIL>"
            help = "silence word symbol"
        "--edge-silprob"
            default = 0.8
            arg_type = Float64
            help = "probability of having a silence at the edge of the utterance"
        "--between-silprob"
            default = 0.1
            arg_type = Float64
            help = "probability of having a silence between two words"
        "--ngram-order"
            default = 3
            arg_type = Int
            help = "ngram order of the language model"
        "hmms"
            required = true
            help = "HMM FSMs in JLD2 format"
        "lexicon"
            required = true
            help = "lexicon text file"
        "denfsm"
            required = true
            help = "output denominator fsm in JLD2 format"
    end
    s.description = """
    Build a denominator graph based on a phonetic language model.
    This script read the text from its standard input.
    """
    parse_args(s)
end

args = parse_commandline()
main(args)
