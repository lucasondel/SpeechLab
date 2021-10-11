### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ d0aeac70-e8b3-4765-9483-e1cbb6233793
begin
	using Pkg
	Pkg.activate("../../")

	md"""
	# Lattice-Free MMI part I: Graphs preparation
	*[Lucas Ondel](https://lucasondel.github.io/), September 2021*

	This is the first step of the Lattice-Free MMI training recipe: preparing the graphs needed by the MMI objective function.
	"""
end

# ╔═╡ e313e3c6-1ba5-11ec-2657-9574ba676d47
begin
	using JLD2
	using MarkovModels
	using PlutoUI
	using Random
	using TOML
end

# ╔═╡ 605fd496-1707-4ce8-84f3-dc2dabe13d4e
TableOfContents()

# ╔═╡ 3bf681b3-9949-4a94-824b-b2f7416751bf
md"""
## Setup
"""

# ╔═╡ ad9289fd-3d73-443d-8a6d-79edcf7d8c0a
md"""
We use the [MarkovModels](https://github.com/lucasondel/MarkovModels.jl) package to build our graphs.
"""

# ╔═╡ b248cbdb-93ac-4e40-92ed-1cbd6f0b8f89
md"""
We load the configuration file of the experiment. By default, we look for the file  `config.toml` in the root directory (i.e. the directory containing this notebook). Alternatively, when calling this notebook as a julia script, you can specify another configuration file by setting the environment variable `SPEECHLAB_LFMMI_CONFIG=/path/to/file`.
"""

# ╔═╡ fb1c9103-ad50-4eeb-b142-ff74f55170c3
rootdir = @__DIR__

# ╔═╡ d8c0ac69-6753-4eb2-bf35-d6a3744eb261
configfile = get(ENV, "SPEECHLAB_LFMMI_CONFIG", joinpath(rootdir, "config.toml"))

# ╔═╡ c16e546d-f83a-4094-bf0c-31457bef0ef5
begin
	@info "Reading configuration from $configfile."
	config = TOML.parsefile(configfile)
end

# ╔═╡ ce7861dd-3868-4fc5-b5c8-339ab1a4981f
md"""
## Input
To build the graphs, we need a *lexicon*, the *list of acoustic units* (i.e. phones), the *topology for each unit* and the *textual transcription* of the training and development set.

### Lexicon
The lexicon contains the phonetic pronunciation of each word of the vocabulary. It has the following format:
```
word1  a b c
word2  e d f
...
```
where `a`, `b`, `c`... are phones in a given language.
!!! note
	A word can have multiple pronunciations. In this case, you can write them on separate lines in the lexicon such as:
	```
	word1 h e l o
	word1 h a l o
    ...
	```

### Units
The list of units is stored in file formatted as:
```
unit1  [tag1 tag2 ...]
unit2  [tag1 tag2 ...]
...
```
Tags are optional user-defined categories. They allow to have unit specific topologies. Here is an example of a units file for the English language:
```
SIL     nonspeech-unit  silence
NSN     nonspeech-unit  nonspoken-noise
SPN     speech-unit     spoken-noise
AA      speech-unit     phone
AE      speech-unit     phone
AH      speech-unit     phone
AO      speech-unit     phone
AW      speech-unit     phone
AY      speech-unit     phone
B       speech-unit     phone
```

### HMM topologies
For each unit, we specify the HMM topology. The path (relative to the root directory) of the topology file is given in in the configuration file by the key <graphs.hmm_topologies>. In this file, you need to specify the HMM states and their arcs. For instance:
```TOML
states = [
	{ initweight = 1.0, finalweight = 0.5 },
	{ initweight = 0.0, finalweight = 0.5 },
]

arcs = [
	{ src = 1, dest = 2, weight = 0.5 },
	{ src = 2, dest = 2, weight = 0.5 },
]
```
If you want to have a particular topology for a given units, you can use the tags specified in the units file. For instance, let's say we have the following units file:
```
SIL     nonspeech-unit  silence
NSN     nonspeech-unit  nonspoken-noise
AA      speech-unit     vowel
AE      speech-unit     vowel
AE      speech-unit     consonant
```
Then we can have different topologies given the category tags.
```TOML

# This is for units that don't match specific categories.
states = [
	{ initweight = 1.0, finalweight = 0.5 },
	{ initweight = 0.0, finalweight = 0.5 },
]

arcs = [
	{ src = 1, dest = 2, weight = 0.5 },
	{ src = 2, dest = 2, weight = 0.5 },
]

# For all nonspeech-units
[nonspeech-units]
states = [
	{ initweight = 1.0, finalweight = 0.5 },
]

arcs = [
	{ src = 1, dest = 1, weight = 0.5 },
]

# For vowels
[speech-units.vowels]
states = [
	{ initweight = 1.0, finalweight = 0.0 },
	{ initweight = 0.0, finalweight = 0.33 },
	{ initweight = 0.0, finalweight = 0.5 },
]

arcs = [
	{ src = 1, dest = 2, weight = 0.5 },
	{ src = 2, dest = 2, weight = 0.33 },
	{ src = 2, dest = 3, weight = 0.33 },
  	{ src = 3, dest = 3, weight = 0.5 },
]
```


### Transcription
The text transcription simply a text file with one line per utterance:
```
utt1  w1 w2 w3 ...
utt2  w1 w2 w3 ...
...
```

We expect these different files to be organized as:
```
<dataset.dir>/
+-- <dataset.name>/
|   +-- lang/
|   |   +-- units
|   |   +-- lexicon
|   +-- dev/
|   |   +-- text
|   +-- train/
|   |   +-- text
```
The keys `<dataset.dir> and <dataset.name> are taken from the configuration file.
!!! note
	`<dataset.dir>` should be an absolute directory.
"""

# ╔═╡ a55eaabf-cb79-4130-affa-a64c3bce1526
lexiconfile = joinpath(config["dataset"]["dir"], config["dataset"]["name"],
					   "lang", "lexicon")

# ╔═╡ 537679cc-3104-4bbb-aab8-21d80ed03ced
unitsfile = joinpath(config["dataset"]["dir"], config["dataset"]["name"],
					 "lang", "units")

# ╔═╡ 43c0ce2c-c28d-4b77-a2ac-75edacc8774f
topofile = joinpath(rootdir, config["graphs"]["hmm_topologies"])

# ╔═╡ 8bd0b6af-9aff-43e7-a06f-934b2525d11a
begin
	@info "Reading topologies from $topofile."
	topologies = TOML.parsefile(topofile)
end

# ╔═╡ cf5fc8b6-e14d-49b2-9c7d-c8e58e80c118
md"""
We have two options to generate the graphs: either we build the graph by "sampling" an alignment sequence or we build the graph of all possible sequence. The generating mode is set is decided by the key `<graphs.sample>`.
"""

# ╔═╡ 6384a034-32ca-4e9e-8a20-d7b044868cca
sample = config["graphs"]["sample"]

# ╔═╡ d6e65144-247f-4c77-b22d-4e59f2433201
train_trans = joinpath(config["dataset"]["dir"], config["dataset"]["name"],
					   "train", "text")

# ╔═╡ 873dd558-f3fa-494f-8fe5-4e353244a44e
dev_trans = joinpath(config["dataset"]["dir"], config["dataset"]["name"],
					   "dev", "text")

# ╔═╡ 59e08585-dff3-4390-a5e8-eb29ffa5b3c2
md"""
## Output
This notebook will generate:
  1. the mapping between the HMMs' states and the pdf-id.
  2. the alignment (or numerator) graphs, one for each utterance.
  3. the phonotactic language model (or denominator) graph shared by all utterances.
The output will have the following structure:
```
<rootdir>/
+-- <graphs.dir>/
|   +-- <dataset.name>/
|   |   +-- denominator_fsm.jld2
|   |   +-- dev_alignments_fsms.jld2
|   |   +-- pdfid_mapping.jld2
|   |   +-- train_alignments_fsms.jld2
```
The keys `<graphs.dir>` and `<dataset.name>` are taken from the configuration file. If it doesn't exists, the path `<rootdir>/<graphs.dir>/<dataset.name>` will be created.
"""

# ╔═╡ 27f7f58c-b27d-49fe-808d-b6028b8825f4
graphsdir = joinpath(rootdir, config["graphs"]["dir"], config["dataset"]["name"])

# ╔═╡ 11db08d7-b24f-4892-8dc4-ee4ff41eef72
mkpath(graphsdir)

# ╔═╡ 8dad4f95-da50-420a-9a80-847374567180
md"""
## HMMs

For each units of the language, we generate a HMM.
"""

# ╔═╡ 3ce30988-b86f-4405-bf9c-814962a19943
hmmfile = joinpath(graphsdir, "hmms.jld2")

# ╔═╡ d372e6e8-8126-42b6-a88e-f0cc5999ad56
md"""
## Lexicon

Now we convert text input lexicon into a FSM-based representation.
"""

# ╔═╡ 6327eb1c-61a8-4bcf-812c-9c5b05707b50
md"""
## Numerator graphs

We create the alignment graphs that will be the numerator graphs of the LF-MMI
loss function. In the mean time, we estimate a n-gram language model for the
denominator graph.

The parameters are:
  * `sil_sym` the silence word symbol
  * `oov_sym` the special symbol to indicate out-of-vocabulary words
  * `between_silprob` the probability to have silence between word
  * `edge_silprob` the probability to have silence at the edge of an utterance
  * `ngram_order` the order of the n-gram language model
"""

# ╔═╡ 5cb1f14f-521e-4378-8089-9b6a546e0594
sil_sym = config["graphs"]["sil_sym"]

# ╔═╡ b1d8df0f-62ba-40c3-ac04-02a2bff6af97
oov_sym = config["graphs"]["oov_sym"]

# ╔═╡ 1ce9985f-79c6-4de4-9fb1-66f9f70c64f3
between_silprob = config["graphs"]["between_silprob"]

# ╔═╡ 0ea4bd97-676b-4c22-ac98-b67c2c052324
edge_silprob = config["graphs"]["edge_silprob"]

# ╔═╡ a7dbdbc1-8397-426b-be12-973dfe812346
ngram_order = config["graphs"]["ngram_order"]

# ╔═╡ f89f0d32-834e-4ff0-9cff-125e159c31b4
train_alifile = joinpath(graphsdir, "train_alignments_fsms.jld2")

# ╔═╡ b0eadb5b-7601-4181-9c9f-7692c5be404d
dev_alifile = joinpath(graphsdir, "dev_alignments_fsms.jld2")

# ╔═╡ 78d343e0-e66c-4e20-a646-47d4ffa5253f
function sample_ali(sentence, silsym, between_silprob, edge_silprob)
	retval = []
	if rand() < edge_silprob push!(retval, silsym) end
	for (i, word) in enumerate(sentence)
		push!(retval, word)
		if i < length(sentence) && rand() < between_silprob 
			push!(retval, silsym) 
		end
	end
	if rand() < edge_silprob push!(retval, silsym) end
	retval
end

# ╔═╡ 789c536c-edc5-44f7-9105-27db55b63cbc
md"""
## Denominator graph

We use the n-gram counts from the previous step to build a phonotactic language model FSM. This language model has no back-off.
"""

# ╔═╡ 329f8e51-eb9a-48ba-ac84-ecc4bc1a6dc4
 den_fsmfile = joinpath(graphsdir, "denominator_fsm.jld2")

# ╔═╡ e1a64ab5-72c6-4a50-8e20-a472a1159021
md"""
## Utilities
"""

# ╔═╡ 3d4d32a3-b633-4b54-abae-3c169c0c8c4c
md"""
### Loading data
"""

# ╔═╡ c067a34d-1a5c-4134-96fe-9d97f3f5bf42
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

# ╔═╡ bd3301dc-9806-4236-9507-c0290bd5e819
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

# ╔═╡ f4a8a935-d6bf-4778-891e-1b4c79d48930
md"""
### Building FSMs
"""

# ╔═╡ c9895639-fabe-4727-9478-e919ab3fbb7e
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

# ╔═╡ 9bf21a5d-3999-44cb-9556-e625c7126e3e
function makehmms(topologies, unitsfile)
	HMMs = Dict()
	pdfid_mapping = Dict()
	units = loadunits(unitsfile)
	 pdfid = 1
	for unit_tuple in units
		unit, categories = unit_tuple[1], unit_tuple[2:end]
		topo = get_topology(topologies, categories)
		(HMMs[unit], next_pdfid) = makehmm!(pdfid_mapping, unit, topo, pdfid)
		pdfid = next_pdfid
	end
	HMMs, pdfid_mapping
end

# ╔═╡ 532d1672-d160-4103-b3a6-6b951faa221f
begin
	HMMs, pdfid_mapping = makehmms(topologies, unitsfile)
	@info "Created $(length(HMMs)) HMMs."
end

# ╔═╡ c3f3e900-c35a-4386-8ba5-e435387941d2
save(joinpath(graphsdir, "pdfid_mapping.jld2"), "pdfid_mapping", Dict(pdfid_mapping))

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
function makelexicon(lexiconfile; sample=false)
	lexicon = Dict()
	pronuns = loadlexicon(lexiconfile)

	# Note: this loop could be parallelized to deal with large lexicon.
	for word in keys(pronuns)
		@debug "Building pronunciation for $word."
		fsms = [LinearFSM(pronun) for pronun in pronuns[word]]
		fsm = sample ? fsms[rand(1:length(fsms))] : union(fsms...)
		lexicon[word] = fsm |> minimize
	end
	lexicon
end

# ╔═╡ dca35409-d9fc-45c9-a826-87f262189211
begin
	@info "Building the lexicon."
	lexicon = makelexicon(lexiconfile; sample)
	@info "Build a pronunciation FSM for $(length(lexicon)) entries."
end

# ╔═╡ 88fff67c-5995-44ca-9a70-d493de00959d
if "HELLO" in keys(lexicon) lexicon["HELLO"] end

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

# ╔═╡ 0dab16d8-98f4-4e19-8663-8e667c1b9d41
function AlignmentFSM(sentence, sil_sym, between_silprob, edge_silprob)
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

# ╔═╡ d1c050fe-f98d-42a1-8ec5-10196811d262
md"""
### Counting n-grams
"""

# ╔═╡ 8dd44223-0a2d-4eb4-9a15-20654ec1ab76
function enumerate_labels(fsm::AbstractFSM{SF}, state, ngram_order,
						  expand_init = false) where SF
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

# ╔═╡ 242f61df-22b1-43ae-b638-870d98aa8dce
function count_ngrams(fsm::AbstractFSM, ngram_order)
	seqs = []
	for state in states(fsm)
		for t in enumerate_labels(fsm, state, ngram_order, true)
			push!(seqs, t)
		end
	end
	seqs
end

# ╔═╡ 63374224-5ad9-4927-a299-389665d061ac
function makenums(fout, trans, lexicon, sil_sym, oov_sym, between_silprob,
				  edge_silprob, ngram_order; njobs, sample=false)

	ch_lines = Channel(100)
	ch_numfsms = Channel(100)
	ch_ngrams = Channel(100)

	Threads.@spawn open(trans, "r") do f
        for line in eachline(f)
		    put!(ch_lines, line)
        end
		close(ch_lines)
	end

	function do_job()
		for line in ch_lines
			tokens = split(line)
			uttid = tokens[1]

			@debug "Creating alignment graph for utterance $uttid."

			# Replace unknown words with the OOV symbol.
			sentence = [word ∈ keys(lexicon) ? word : oov_sym
						for word in tokens[2:end]]

			# Word level FSM.
			trans_wrd = if sample
				seq = sample_ali(sentence, sil_sym, between_silprob, edge_silprob)
				LinearFSM(seq)
			else
				AlignmentFSM(sentence, sil_sym, between_silprob,
							 edge_silprob)
			end

			# Phone level FSM.
			trans_phn = HierarchicalFSM(trans_wrd, lexicon)

			# We count the n-grams that will be used later
			# to build the denominator graph.
			rawcounts = count_ngrams(trans_phn, ngram_order)
			for (tup, ws) in rawcounts
				(src, dest) = getstates(tup, ngram_order)
				put!(ch_ngrams, (src, dest, ws))

			end

			# Alignment FSM.
			ali = HierarchicalFSM(trans_phn, HMMs)

			# Compile the FSM to a matrix format.
			matfsm = MatrixFSM(ali, pdfid_mapping, t -> (t[2], t[3]))

			put!(ch_numfsms, (uttid, matfsm))
		end
		close(ch_numfsms)
		close(ch_ngrams)
	end

	Threads.@spawn do_job()
	# for jobid in 1:njobs
	# 	Threads.@spawn do_job()
	# end

	ngram_counts = Dict()
	Threads.@spawn for (src, dest, ws) in ch_ngrams
		c = get(ngram_counts, (src, dest),
			    (zero(ws[1]), zero(ws[1]), zero(ws[1])))
		ngram_counts[(src,dest)] = c .+  ws
	end


	for (uttid, numfsm) in ch_numfsms
		fout[uttid] = numfsm
	end

	ngram_counts
end

# ╔═╡ 6cfb9bd9-e5e9-4983-be9c-4f96711590e1
ngram_counts = jldopen(train_alifile, "w") do fout
	@info "Generating alignments FSMs for the train set."
	makenums(fout, train_trans, lexicon, sil_sym, oov_sym, between_silprob,
			 edge_silprob, ngram_order; njobs = Threads.nthreads(), sample)
end

# ╔═╡ ee118372-9d16-4de6-84b2-9900036f2f8f
jldopen(den_fsmfile, "w") do f
	@info "Creating the $(ngram_order)-gram phonotactic LM."
	fsm = LanguageModelFSM(ngram_counts)
	fsm = HierarchicalFSM(fsm, HMMs)
	f["fsm"] = MatrixFSM(fsm, pdfid_mapping, t -> (t[end-1], t[end]))
end;

# ╔═╡ b64876f4-db49-493a-bf0e-85096dfe3f88
jldopen(dev_alifile, "w") do fout
	@info "Generating alignments FSMs for the dev set."
	makenums(fout, dev_trans, lexicon, sil_sym, oov_sym, between_silprob,
			 edge_silprob, ngram_order; njobs=Threads.nthreads(), sample)
end

# ╔═╡ Cell order:
# ╟─d0aeac70-e8b3-4765-9483-e1cbb6233793
# ╠═605fd496-1707-4ce8-84f3-dc2dabe13d4e
# ╟─3bf681b3-9949-4a94-824b-b2f7416751bf
# ╟─ad9289fd-3d73-443d-8a6d-79edcf7d8c0a
# ╠═e313e3c6-1ba5-11ec-2657-9574ba676d47
# ╟─b248cbdb-93ac-4e40-92ed-1cbd6f0b8f89
# ╠═fb1c9103-ad50-4eeb-b142-ff74f55170c3
# ╠═d8c0ac69-6753-4eb2-bf35-d6a3744eb261
# ╠═c16e546d-f83a-4094-bf0c-31457bef0ef5
# ╟─ce7861dd-3868-4fc5-b5c8-339ab1a4981f
# ╠═a55eaabf-cb79-4130-affa-a64c3bce1526
# ╠═537679cc-3104-4bbb-aab8-21d80ed03ced
# ╠═43c0ce2c-c28d-4b77-a2ac-75edacc8774f
# ╠═8bd0b6af-9aff-43e7-a06f-934b2525d11a
# ╟─cf5fc8b6-e14d-49b2-9c7d-c8e58e80c118
# ╠═6384a034-32ca-4e9e-8a20-d7b044868cca
# ╠═d6e65144-247f-4c77-b22d-4e59f2433201
# ╠═873dd558-f3fa-494f-8fe5-4e353244a44e
# ╟─59e08585-dff3-4390-a5e8-eb29ffa5b3c2
# ╠═27f7f58c-b27d-49fe-808d-b6028b8825f4
# ╠═11db08d7-b24f-4892-8dc4-ee4ff41eef72
# ╟─8dad4f95-da50-420a-9a80-847374567180
# ╠═3ce30988-b86f-4405-bf9c-814962a19943
# ╠═9bf21a5d-3999-44cb-9556-e625c7126e3e
# ╠═532d1672-d160-4103-b3a6-6b951faa221f
# ╠═c3f3e900-c35a-4386-8ba5-e435387941d2
# ╟─d372e6e8-8126-42b6-a88e-f0cc5999ad56
# ╠═30f79da9-8f89-4690-a45e-9d8f0063b6e8
# ╠═dca35409-d9fc-45c9-a826-87f262189211
# ╠═88fff67c-5995-44ca-9a70-d493de00959d
# ╟─6327eb1c-61a8-4bcf-812c-9c5b05707b50
# ╠═5cb1f14f-521e-4378-8089-9b6a546e0594
# ╠═b1d8df0f-62ba-40c3-ac04-02a2bff6af97
# ╠═1ce9985f-79c6-4de4-9fb1-66f9f70c64f3
# ╠═0ea4bd97-676b-4c22-ac98-b67c2c052324
# ╠═a7dbdbc1-8397-426b-be12-973dfe812346
# ╠═f89f0d32-834e-4ff0-9cff-125e159c31b4
# ╠═b0eadb5b-7601-4181-9c9f-7692c5be404d
# ╠═78d343e0-e66c-4e20-a646-47d4ffa5253f
# ╠═63374224-5ad9-4927-a299-389665d061ac
# ╠═6cfb9bd9-e5e9-4983-be9c-4f96711590e1
# ╠═b64876f4-db49-493a-bf0e-85096dfe3f88
# ╟─789c536c-edc5-44f7-9105-27db55b63cbc
# ╠═329f8e51-eb9a-48ba-ac84-ecc4bc1a6dc4
# ╠═ee118372-9d16-4de6-84b2-9900036f2f8f
# ╟─e1a64ab5-72c6-4a50-8e20-a472a1159021
# ╟─3d4d32a3-b633-4b54-abae-3c169c0c8c4c
# ╠═c067a34d-1a5c-4134-96fe-9d97f3f5bf42
# ╠═bd3301dc-9806-4236-9507-c0290bd5e819
# ╟─4dd7c660-34bb-493e-8e6c-e23c45eb2687
# ╟─f4a8a935-d6bf-4778-891e-1b4c79d48930
# ╠═c9895639-fabe-4727-9478-e919ab3fbb7e
# ╠═ab1d6284-5c71-4e46-b5b1-72c75498a7b9
# ╟─0e043217-7087-47f1-910b-937322c78159
# ╠═0dab16d8-98f4-4e19-8663-8e667c1b9d41
# ╠═cac1b00e-4eaa-4057-8cbe-3571a15c8fa9
# ╟─d1c050fe-f98d-42a1-8ec5-10196811d262
# ╠═8dd44223-0a2d-4eb4-9a15-20654ec1ab76
# ╠═242f61df-22b1-43ae-b638-870d98aa8dce
