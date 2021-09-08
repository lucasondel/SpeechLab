# SPD-License-Identifier: MIT

using ArgParse
using JLD2
using MarkovModels
using TOML

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "topology"
            required = true
            help = "hmm topology in TOML format"
        "units"
            required = true
            help = "list of units with their categories"
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

function loadunits(file)
    units, categories = [], []
    open(file, "r") do f
        for line in eachline(f)
            tokens = split(line)
            push!(units, tokens[1])
            push!(categories, tokens[2:end])
        end
    end
    units, categories
end

function get_unit_topo(topo, category)
    i = 1
    while i <= length(category) && category[i] ∈ keys(topo)
        topo = topo[category[i]]
        i += 1
    end
    topo
end

function makehmm!(pdfid_mapping, unit, topo, pdfid)
    SF = LogSemifield{Float32}
    fsm = VectorFSM{SF}()

    states = Dict()
    for (i, state) in enumerate(topo["states"])
        initweight = SF(log(state["initweight"]))
        finalweight = SF(log(state["finalweight"]))
        s = addstate!(fsm, i; initweight, finalweight)
        states[i] = s
        pdfid_mapping[(unit, i)] = pdfid
        pdfid += 1
    end

    for arc in topo["arcs"]
        addarc!(fsm, states[arc["src"]], states[arc["dest"]],
                SF(log(arc["weight"])))
    end

    fsm |> renormalize, pdfid
end

function main(args)
    topo = TOML.parsefile(args["topology"])

    units, categories = loadunits(args["units"])
    hmms = Dict()
    pdfid_mapping = Dict()
    pdfid = 1
    for (unit, category) in zip(units, categories)
        unit_topo = get_unit_topo(topo, category)
        hmms[unit], pdfid = makehmm!(pdfid_mapping, unit, topo, pdfid)
    end

    data = Dict(
        "units" => hmms,
        "pdfid_mapping" => pdfid_mapping
    )
    save(args["hmms"], data)
end

args = parse_commandline()
main(args)

