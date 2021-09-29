### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ e313e3c6-1ba5-11ec-2657-9574ba676d47
begin
	using Pkg
	Pkg.activate("../../")
	
	using MarkovModels
	using JLD2
	using Random
	using TOML
	
	using PlutoUI
	TableOfContents()
end

# ╔═╡ d0aeac70-e8b3-4765-9483-e1cbb6233793
md"""
# Graph preparation

In this notebook, we build the numerator and denominator graphs of the LF-MMI
objective function.
"""

# ╔═╡ 3bf681b3-9949-4a94-824b-b2f7416751bf
md"""
## Global Settings
"""

# ╔═╡ ca219d0b-112e-456f-abde-2d08d41225f9
begin
	config = TOML.parsefile("/people/ondel/SpeechLab/recipes/lfmmi/config.toml")
	
	dataset = config["dataset"]
	datasetdir = joinpath(config["datasetsdir"], dataset)
	expdir = joinpath(config["outdir"], config["dataset"])
	graphsdir = joinpath(expdir, "graphs")	
	topologies = TOML.parsefile(joinpath(config["rootdir"], 
								config["graphs"]["hmm_topologies"]))
	ngram_order= config["graphs"]["ngram_order"]
	
	mkpath(graphsdir)
end;

# ╔═╡ 8dad4f95-da50-420a-9a80-847374567180
md"""
## HMMs

Prepare the units' HMMs.
"""

# ╔═╡ 3ce30988-b86f-4405-bf9c-814962a19943
hmmfile = joinpath(graphsdir, "hmms.jld2")

# ╔═╡ d372e6e8-8126-42b6-a88e-f0cc5999ad56
md"""
## Lexicon

Represent a lexicon as a collection of FSM.
"""

# ╔═╡ 6327eb1c-61a8-4bcf-812c-9c5b05707b50
md"""
## Numerator graphs

The numerator graphs correspond to the alignment graphs. In the mean time, we will estimate the $(ngram_order)-gram language model. 
"""

# ╔═╡ 0ea4bd97-676b-4c22-ac98-b67c2c052324
begin
	sil_sym = config["graphs"]["sil_sym"]
	oov_sym = config["graphs"]["oov_sym"]
	between_silprob = config["graphs"]["between_silprob"]
	edge_silprob = config["graphs"]["edge_silprob"]
end;

# ╔═╡ f89f0d32-834e-4ff0-9cff-125e159c31b4
train_alifile = joinpath(graphsdir, "train_alignments_fsms.jld2")

# ╔═╡ b0eadb5b-7601-4181-9c9f-7692c5be404d
dev_alifile = joinpath(graphsdir, "dev_alignments_fsms.jld2")

# ╔═╡ 90f50f32-bd76-4341-b1fe-3285ef948623
jldopen(train_alifile, "r") do f
	f["lbi-118-47824-0026"]
end

# ╔═╡ 83aa75c6-4f72-4e87-9ec7-c237371b6d1b
"pay him will you Jenkins thanks"

# ╔═╡ 789c536c-edc5-44f7-9105-27db55b63cbc
md"""
## Denominator graph

The denominator graph is a $(ngram_order)-gram language model without any backoff.
"""

# ╔═╡ 329f8e51-eb9a-48ba-ac84-ecc4bc1a6dc4
 den_fsmfile = joinpath(graphsdir, "denominator_fsm.jld2")

# ╔═╡ e1a64ab5-72c6-4a50-8e20-a472a1159021
md"""
## Utility Functions
"""

# ╔═╡ c067a34d-1a5c-4134-96fe-9d97f3f5bf42
begin
	""" 
		loadunits(file)
	
	Load the units from a text file. The file should be formatted as:
	```
	unit1 [category1 category2...]
	unit2 [category1 category2...]
	```
	
	!!! note
		Each unit can have a different number of categories.
	
	"""
	loadunits
	
	function loadunits(file)
		units = []
		open(file, "r") do f
			for line in eachline(f)
				tokens = split(line)
				push!(units, tuple(tokens...))
			end
		end
		units
	end
end

# ╔═╡ bd3301dc-9806-4236-9507-c0290bd5e819
begin
	""" 
		loadlexicon(file)
	
	Load the lexicon from a text file. The file should be formatted as:
	```
	word1 a b c...
	word1 a c d...
	word2 e f d 
	```
	
	!!! note
		A word can have multiple pronunciation.
	
	"""
	loadlexicon
	
	function loadlexicon(file)
		pronuns = Dict()
		open(file, "r") do f
			for line in eachline(f)
				tokens = split(line)
				word, pronun = tokens[1], tokens[2:end]
				plist = get(pronuns, word, [])
				push!(plist, pronun)
				pronuns[word] = plist
			end
		end
		pronuns
	end
end

# ╔═╡ 4dd7c660-34bb-493e-8e6c-e23c45eb2687
begin
	"""
		get_topology(topologies, unit_tuple)
	
	Get the topology for the given unit. `category` is a sequence
	of category tags. These tags will be use to select the topology.
	"""
	get_topology
	
	function get_topology(topologies, categories)
		i = 2 
		topo = topologies
		while i <= length(categories) && categories[i] ∈ keys(topologies)
			topo = topo[categories[i]]
			i += 1
		end
		topo
	end
end

# ╔═╡ c9895639-fabe-4727-9478-e919ab3fbb7e
begin
	"""
		makehmm!(pdfid_mapping, unit, topo, pdfid) -> (fsm, next_pdfid)
	
	Build a HMM graph. `pdfid_mapping` is a dictionary-like object where the 
	created pdf ids will be stored (the keys are the state label). The 
	function returns a FSM and the next pdf id to use. 
	"""
	makehmm!
	
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
			src, dest = states[arc["src"]], states[arc["dest"]]
			weight = SF(log(arc["weight"]))
			addarc!(fsm, src, dest, weight)
		end
		
		fsm, pdfid
	end
end

# ╔═╡ 9bf21a5d-3999-44cb-9556-e625c7126e3e
begin
	HMMs = Dict()
	pdfid_mapping = Dict()
	local units = loadunits(joinpath(datasetdir, "lang", "units"))
	local pdfid = 1
	for unit_tuple in units
		unit, categories = unit_tuple[1], unit_tuple[2:end]
		topo = get_topology(topologies, categories)
		(HMMs[unit], next_pdfid) = makehmm!(pdfid_mapping, unit, topo, pdfid)
		pdfid = next_pdfid
	end
end

# ╔═╡ c3f3e900-c35a-4386-8ba5-e435387941d2
save(joinpath(graphsdir, "pdfid_mapping.jld2"), 
	 Dict("pdfid_mapping" => pdfid_mapping))

# ╔═╡ ab1d6284-5c71-4e46-b5b1-72c75498a7b9
begin
	"""
		LinearFSM(seq)
	
	Create a linear FSM from a sequence of labels.
	"""
	LinearFSM
	
	function LinearFSM(seq)
		SF = LogSemifield{Float32}
		fsm = VectorFSM{SF}()
		
		prev = nothing
		for (i, label) in enumerate(seq)
			initweight = i == 1 ? one(SF) : zero(SF)
			finalweight = i == length(seq) ? one(SF) : zero(SF)
			s = addstate!(fsm, label; initweight, finalweight)
			i > 1 && addarc!(fsm, prev, s)
			prev = s
		end
		fsm
	end
end

# ╔═╡ 30f79da9-8f89-4690-a45e-9d8f0063b6e8
begin
	lexicon = Dict()
	local pronuns = loadlexicon(joinpath(datasetdir, "lang", "lexicon"))
	for word in keys(pronuns)
		fsms = [LinearFSM(pronun) for pronun in pronuns[word]]
		fsm = union(fsms...)
		lexicon[word] = union(fsms...) |> minimize 
	end
end

# ╔═╡ 5c9c5de9-9b7e-44a4-91ad-a7566803fed5
lexicon

# ╔═╡ 05b069e9-87c6-4a59-a667-c550c03e633e
lexicon["HELLO"]

# ╔═╡ 0dab16d8-98f4-4e19-8663-8e667c1b9d41
begin
	"""
		alignment_fsm([LogSemifield{Float32}], sentence, sil_sym, between_silprob, edge_silprob)
	
	Create the alignment FSM. 
	"""
	alignment_fsm
	
	function alignment_fsm(sentence, sil_sym, between_silprob, edge_silprob)
		SF = LogSemifield{Float32}
		fsm = VectorFSM{SF}()
		
		# Initial optional silence state.
		prev = addstate!(fsm, sil_sym; 
					     initweight = SF(log(edge_silprob)),
						 finalweight = zero(SF))
		
		for (i, word) in enumerate(sentence)
			initweight = i == 1 ? SF(log(1 - edge_silprob)) : zero(SF)
			finalweight = i == length(sentence) ? SF(log(1 - edge_silprob)) : zero(SF)
			s = addstate!(fsm, word; initweight, finalweight)
			
			
			if i > 1
				silstate = addstate!(fsm, sil_sym)
				addarc!(fsm, prev, silstate, SF(log(between_silprob)))
				addarc!(fsm, prev, s, SF(log(1-between_silprob)))
				addarc!(fsm, silstate, s)
			else
				addarc!(fsm, prev, s)
			end
			
			prev = s
		end
		
		# Final optional silence state.
		silstate = addstate!(fsm, sil_sym;
							 initweight = zero(SF),
							 finalweight = one(SF))
		addarc!(fsm, prev, silstate, SF(log(edge_silprob)))
		
		fsm
	end
end

# ╔═╡ 242f61df-22b1-43ae-b638-870d98aa8dce
begin
	
	function enumerate_labels(fsm::AbstractFSM{SF}, state, ngram_order, expand_init = false) where SF
		retval = []
		if expand_init && isinit(state)
			push!(retval, ((state.label[end],), (state.initweight, zero(SF), zero(SF))))
			for i in 2:ngram_order-1
				push!(retval, enumerate_labels(fsm, state, i)...)
			end
		end
		
		if ngram_order == 1
			push!(retval, ((state.label[end],), (zero(SF), one(SF), state.finalweight)))
			return retval
		end
		
		for a in arcs(fsm, state)
			aw = a.weight
			for (seqs, (iw, w, fw)) in enumerate_labels(fsm, a.dest, ngram_order-1)
				push!(retval, (tuple(state.label[end], seqs...), (iw, aw * w, fw)))
			end
		end
		
		retval
	end
	
	"""
		count_ngrams(fsm, order)
	
	Count the number of occurences of all the `order`-grams in `fsm`.
	"""
	count_ngrams
	
	function count_ngrams(fsm::AbstractFSM, ngram_order)
		seqs = []
		for state in states(fsm)
			for t in enumerate_labels(fsm, state, ngram_order, true)
				push!(seqs, t)
			end
		end
		seqs
	end
end

# ╔═╡ cac1b00e-4eaa-4057-8cbe-3571a15c8fa9
begin
	"""
		getstates(tup, order)
	
	Return the state label (source and destination) of the `order`-ngram
	stored in `tup`.
	"""
	getstates
	
	function getstates(tup, ngram_order)
		if length(tup) == 1
			return (nothing, tup)
		end
		L = length(tup)
		src = tuple(tup[1:min(ngram_order, L)-1]...)
		dest = tuple(tup[max(L-ngram_order+2, 1):end]...)
		src, dest
	end
	
end

# ╔═╡ 63374224-5ad9-4927-a299-389665d061ac
begin
	ngram_counts = Dict()
	for (setname, alifile) in [("train", train_alifile), ("dev", dev_alifile)]
		jldopen(alifile, "w") do fout
			open(joinpath(datasetdir, setname, "trans.wrd"), "r") do f

				for line in collect(eachline(f))
					tokens = split(line)
					uttid = tokens[1]
					
					# if setname == "train" && uttid ≠ "lbi-118-47824-0026"
					# 	continue
					# end

					@info "Processing utterance $uttid in $setname." 

					sentence = [word ∈ keys(lexicon) ? word : oov_sym 
								for word in tokens[2:end]]

					# Word level FSM.
					trans_wrd = alignment_fsm(sentence, sil_sym, between_silprob, 
											  edge_silprob)

					# Phone level FSM.
					trans_phn = HierarchicalFSM(trans_wrd, lexicon)

					# We count the n-grams that will be used later
					# to build the denominator graph.
					if setname == "train"
						rawcounts = count_ngrams(trans_phn, ngram_order)
						for (tup, ws) in rawcounts
							(src, dest) = getstates(tup, ngram_order)
							c = get(
								ngram_counts, 
								(src, dest), 
								(zero(ws[1]), zero(ws[1]), zero(ws[1]))
							)
							ngram_counts[(src,dest)] = c .+  ws
						end
					end

					# Alignment FSM. 
					ali = HierarchicalFSM(trans_phn, HMMs)

					# Compile the FSM to a matrix format.
					matfsm = MatrixFSM(ali, pdfid_mapping, t -> (t[2], t[3]))

					fout[uttid] = matfsm
					
				end
			end
		end
	end
end

# ╔═╡ f2295b6d-e479-48e2-a242-edfe1aca1de2
ngram_counts

# ╔═╡ 0e043217-7087-47f1-910b-937322c78159
begin
	"""
		LanguageModelFSM(ngram_counts)
	
	Create a language model FSM without back-offs.
	"""
	LanguageModelFSM
	
	function LanguageModelFSM(ngram_counts)
		SR = LogSemifield{Float32}
		fsm = VectorFSM{SR}()

		iws = Dict()
		fws = Dict()
		for ((src, dest), (iw, w, fw)) in ngram_counts
			if isnothing(src) iws[dest] = iw end
			if fw ≠ zero(w) fws[dest] = fw end
		end


		smap = Dict()
		for ((src, dest), (_, w, _)) in ngram_counts
			dest ∈ keys(smap) && continue
			initweight = get(iws, dest, zero(w))
			finalweight = get(fws, dest, zero(w))
			smap[dest] = addstate!(fsm, dest; initweight, finalweight)
		end

		for ((src, dest), (iw, w, fw)) in ngram_counts
			isnothing(src) && continue
			addarc!(fsm, smap[src], smap[dest], w)
		end

		fsm |> renormalize
		#fsm
	end
end

# ╔═╡ ee118372-9d16-4de6-84b2-9900036f2f8f
jldopen(den_fsmfile, "w") do f
	# Create the phonotactic LM.
	fsm = LanguageModelFSM(ngram_counts)

	# Compose the LM with the HMMs.
	fsm = HierarchicalFSM(fsm, HMMs)

	# Compile the FSM to a matrix format.
	f["fsm"] = MatrixFSM(fsm, pdfid_mapping, t -> (t[end-1], t[end]))
end;

# ╔═╡ Cell order:
# ╟─d0aeac70-e8b3-4765-9483-e1cbb6233793
# ╠═e313e3c6-1ba5-11ec-2657-9574ba676d47
# ╟─3bf681b3-9949-4a94-824b-b2f7416751bf
# ╠═ca219d0b-112e-456f-abde-2d08d41225f9
# ╟─8dad4f95-da50-420a-9a80-847374567180
# ╠═3ce30988-b86f-4405-bf9c-814962a19943
# ╠═9bf21a5d-3999-44cb-9556-e625c7126e3e
# ╠═c3f3e900-c35a-4386-8ba5-e435387941d2
# ╟─d372e6e8-8126-42b6-a88e-f0cc5999ad56
# ╠═30f79da9-8f89-4690-a45e-9d8f0063b6e8
# ╠═5c9c5de9-9b7e-44a4-91ad-a7566803fed5
# ╠═05b069e9-87c6-4a59-a667-c550c03e633e
# ╟─6327eb1c-61a8-4bcf-812c-9c5b05707b50
# ╠═0ea4bd97-676b-4c22-ac98-b67c2c052324
# ╠═f89f0d32-834e-4ff0-9cff-125e159c31b4
# ╠═b0eadb5b-7601-4181-9c9f-7692c5be404d
# ╠═63374224-5ad9-4927-a299-389665d061ac
# ╠═90f50f32-bd76-4341-b1fe-3285ef948623
# ╠═83aa75c6-4f72-4e87-9ec7-c237371b6d1b
# ╟─789c536c-edc5-44f7-9105-27db55b63cbc
# ╠═329f8e51-eb9a-48ba-ac84-ecc4bc1a6dc4
# ╠═f2295b6d-e479-48e2-a242-edfe1aca1de2
# ╠═ee118372-9d16-4de6-84b2-9900036f2f8f
# ╟─e1a64ab5-72c6-4a50-8e20-a472a1159021
# ╟─c067a34d-1a5c-4134-96fe-9d97f3f5bf42
# ╟─bd3301dc-9806-4236-9507-c0290bd5e819
# ╟─4dd7c660-34bb-493e-8e6c-e23c45eb2687
# ╟─c9895639-fabe-4727-9478-e919ab3fbb7e
# ╟─ab1d6284-5c71-4e46-b5b1-72c75498a7b9
# ╟─0dab16d8-98f4-4e19-8663-8e667c1b9d41
# ╟─242f61df-22b1-43ae-b638-870d98aa8dce
# ╟─cac1b00e-4eaa-4057-8cbe-3571a15c8fa9
# ╟─0e043217-7087-47f1-910b-937322c78159
