### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 4df289c2-b460-49a6-b1b3-dac6c557397d
begin	
	# TODO: remove this lines to use the notebook package manager.
	using Pkg
	Pkg.activate("../../")
	
	md"""
	# Lattice-free MMI training
	[*Lucas Ondel*](https://lucasondel.github.io/index), September 2021

	
	This notebook implements the creation and the training of a *Time-Delay Neural
	Network* (TDNN) with the *Lattice-Free Maximum Mutual Information* (LF-MMI)
	objective function.
	""" 
end

# ╔═╡ ee11cbaf-73c0-45db-84f7-db9eab7b6005
begin	
	using AutoGrad
	using CUDA
	using Dates
	using Knet
	using Logging
	using MarkovModels
	using HDF5
	using JLD2
	using PlutoUI
	using Random
	using TOML
end

# ╔═╡ 6112bcf7-f4fb-428e-acba-aa64a2ddd555
using Plots

# ╔═╡ fc878252-e6cd-4663-ad45-e42cae318ffb
TableOfContents() 

# ╔═╡ 2ac818e7-7c52-4dfb-9f79-519fbce9d651
md"""
## Setup

Depending on your setup, select the CUDA version you want to use. You can also deactivate this cell and let the julia environment decide for itself.
"""

# ╔═╡ 876cffca-c46b-4805-a00f-5ac30d0631fe
ENV["JULIA_CUDA_VERSION"] = "10.2"

# ╔═╡ c10dd5eb-7bbe-4427-8a04-a3219d69942f
md"""
!!! warning
	If you change the CUDA version in the line above, you will need to restart the
	notebook to make the change effective.
"""

# ╔═╡ 850ac54c-b2e8-4e8e-a67a-b83f8d4d5905
md"""
Import the dependencies. Importantly, we use:
  * [MarkovModels](https://github.com/lucasondel/MarkovModels.jl) to implement the LF-MMI function and it's gradient
  * [KNet](https://github.com/denizyuret/Knet.jl) for the neural-network functions
  * [AutoGrad](https://github.com/denizyuret/AutoGrad.jl) for automatic differentiation. 

!!! note
	It is possible to use [Flux](https://github.com/FluxML/Flux.jl)/[Zygote](https://github.com/FluxML/Zygote.jl) 
    for the neural-network/automatic differentiation backend with little modification 
    to this notebook. The most notable changes will be to specify the gradient with 
	Zygote API and to adapt the creation of the network.
"""

# ╔═╡ ad6986d7-71b3-49cc-92e4-dadf42953b19
md"""
For information, we print the CUDA configuration:
"""

# ╔═╡ bbde1167-ac11-4c15-8374-daa3f807cf3f
with_terminal() do
	CUDA.versioninfo()
end

# ╔═╡ 89a40466-0e49-41b4-882a-20c66ad840b7
md"""
We consider the directory containing this notebook to be the root directory of the experiment.
"""

# ╔═╡ 20d81979-4fd8-4b51-9b27-70f54fb2f886
rootdir = @__DIR__

# ╔═╡ 016eabc9-127c-4e12-98ab-8cd5004edfa0
md"""
We use the file named "`config.toml`" in the root directory as the configuration 
file of this experiment. Alternatively, when calling this notebook as a julia script, 
you can specify another file by setting the environment variable 
`SPEECHLAB_LFMMI_CONFIG=/path/to/file`.
"""

# ╔═╡ 104fc433-def8-44b1-b309-13d7683e0b33
config = TOML.parsefile(
	get(ENV, "SPEECHLAB_LFMMI_CONFIG", joinpath(rootdir, "config.toml"))
)

# ╔═╡ 8bf454e4-7707-4184-a9b3-13a9d072576a
md"""
Here is the directory structure of the experiment:

```
<rootdir>/
+-- config.toml
+-- <expdir>/
|   +-- <dataset>/
|   |   +-- graphs/  # This directory is prepared with "buildgraphs.jl".
|   |   |   +-- denominator_fsm.jld2
|   |   |   +-- dev_alignments_fsms.jld2
|   |   |   +-- pdfid_mapping.jld2
|   |   |   +--- train_alignments_fsms.jld2
|   |   +-- train/   # This directory and its content are created by this notebook.
|   |   |   +-- checkpoint.jld2
|   |   |   +-- best.jld2
|   |   |   +-- log.txt
|   |   +-- output/  # This directory and its content are created by this notebook.
|   |   |   +-- test.h5  
```
The keys `<expdir` and `<dataset` are taken from configuration file.
"""

# ╔═╡ 7dbf440a-8550-48fe-869d-850e9cd79656
expdir = joinpath(rootdir, config["expdir"], config["dataset"])

# ╔═╡ 8ab11e2e-af68-43d1-a766-7095719133ed
graphsdir = joinpath(expdir, "graphs")

# ╔═╡ ca199db4-442f-4caf-a84e-1b4bf73e6112
traindir = joinpath(expdir, "train")

# ╔═╡ b9f56cb7-5cea-4143-b4a4-251306bef902
outdir = joinpath(expdir, "output")

# ╔═╡ 71f17a4a-88d5-449a-b0d0-fdee2956c96c
mkpath.([traindir, outdir])

# ╔═╡ 82a88680-a834-49f2-b0ef-e6dbbc645131
md"""
Set the numeric precision in the experiment. 
"""

# ╔═╡ 3d26b72a-65ba-4e6f-9e24-041007b5c413
T = Float32

# ╔═╡ 270cb75f-62ad-4435-9932-3ff37d6db89f
use_gpu = true

# ╔═╡ aa271670-9e13-4838-968c-e15b34e47123
md"""
## Input features

We assume the features to be stored in [HDF5 format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) organized as:
```
<features.dir>/
+-- <dataset>/
|   +-- train/
|   |   +-- <features.name>.h5
|   +-- dev/
|   |   +-- <features.name>.h5
|   +-- test/
|   |   +-- <features.name>.h5
```
where `<features.dir>` and `<features.name>` are read from the configuration file.
"""

# ╔═╡ 8936b2c7-68b6-4845-94b5-6512a1cb16a2
feadir = joinpath(config["features"]["dir"], config["dataset"])

# ╔═╡ 6e8c6723-d6d9-4dbe-acd7-c0575b1f832e
feaname = config["features"]["name"]

# ╔═╡ 994a581f-f5fa-4c1c-b772-8bb6c8f5a0f5
featrain = joinpath(feadir, "train", feaname * ".h5")

# ╔═╡ 6f336813-1d30-4dc4-a298-87019e7b7f36
feadev = joinpath(feadir, "dev", feaname * ".h5")

# ╔═╡ b5aa09be-a254-4b3e-81ed-91fe31b61c0e
featest = joinpath(feadir, "test", feaname * ".h5")

# ╔═╡ d137c3f8-ceb5-4fbb-8e46-72bdb397d072
md"""
## Model creation

We use a TDNN model with ReLU activations. The input dimension corresponds to the input features dimension.
"""

# ╔═╡ f3e1352d-9b13-4378-9b5d-6ff7f65a4005
indim = h5open(featrain, "r") do f
	uttid, _ = iterate(keys(f))
	size(read(f[uttid]), 1)
end

# ╔═╡ 69f0e6a3-e620-4b55-951f-d32488a48fc5
md"""
The output dimension corresponds to the number of pdf-ids.
"""

# ╔═╡ 110a9455-f3f4-49bf-9d26-f6ed94440076
outdim = jldopen(joinpath(graphsdir, "pdfid_mapping.jld2"), "r") do f
	maximum( t -> t[2], f["pdfid_mapping"])
end

# ╔═╡ a2d14acd-fed2-42b9-85dd-77c5e04b982d
md"""

We now address the creation of the model. Two possible scenarii:
  * a checkpoint from a previous (unfinished) training exists at `.../train/checkpoint.jld2`, then we simply load the model from this file.
  * there is no checkpoint, we create a new model.
"""

# ╔═╡ 0b7d14dc-d580-4e5d-96c7-fd185deed2ed
ckpt_path = joinpath(traindir, "checkpoint.jld2")

# ╔═╡ 67fd7845-47ca-4b3a-9b78-68db1586ea9e
md"""
## Training
"""

# ╔═╡ c8dc633e-81a6-41ee-b61f-90b26dc04494
md"""
We define here the main function of the training. The input arguments are:

|Argument       | Description                                           |
|:--------------|:------------------------------------------------------|
| `model`       | Model to train                                        |
| `trainconfig` | Training part of the configuration of the experiment  |
| `trainfea`    | Training features archive                             | 
| `devfea`      | Development features archive                          | 
| `den_fsm`     | Phonotactic language model FSM for the denominator    | 
| `use_gpu`     |  Whether or not to use a GPU                          | 
| `logfile`     | Log file where to report the progress of the training | 
"""

# ╔═╡ 357e7c9d-9537-46f3-9153-b0bbccfd93b8
jldopen(joinpath(graphsdir, "denominator_fsm.jld2"), "r") do f
	den_fsm = f["fsm"];
	#fsm # |> MarkovModels.gpu
end # <- Don't remove this ';' as this object is too big to be displayed.

# ╔═╡ 8b9b85ef-2419-4581-acca-31c033405587


# ╔═╡ b95c2872-2dd4-4de5-ad81-c7b818ae21e0
md"""
### Neural Network API

Since KNet provide a rather low-level API to build neural network. We define a few utilities that will make things easier for people accustomed to pytorch/tensorflow.
"""

# ╔═╡ 446c8edb-e129-4f2f-a676-1fcf502e24d6
begin
	"""
		reallocate(obj, atype)
	
	Returns a copy of `obj'` with its buffers of type `atype`. 
	This function is used to move the model to/from CPU/GPU.
	"""
	reallocate
	
	"""
		gpu(obj)

	Return a copy `obj` allocated on GPU.
	"""
	gpu
	
	"""
		cpu(obj)

	Return a copy `obj` allocated on CPU.
	"""
	cpu
	
	"""
		trainmode!(model)
	
	Set the model on "train mode" (i.e. activate dropout, batch-norm, ...).
	"""
	trainmode!
	
	"""
		testmode!(model)
	
	Set the model on "test mode". 
	"""
	testmode!
	
	trainmode!(x) = x
	testmode!(x) = x
	# By default, the function does nothing but to copy the object.
	reallocate(obj, atype) = deepcopy(obj)
	gpu(obj) = reallocate(obj, CuArray{T})
	cpu(obj) = reallocated(obj, Array{T})
	
	md"""
	#### Helper functions
	"""
end

# ╔═╡ 007c7b74-59f4-4154-9679-46169624aeea
begin
	"""
		struct PermuteDims
			perm
		end
	
	Neural network layer that permutes the dimension of the input layer.
	"""
	struct PermuteDims perm end
	(p::PermuteDims)(X) = permutedims(X, p.perm)
	Base.show(io::IO, p::PermuteDims) = print(io, "PermuteDims($(p.perm))")
	
	"""
		struct AddExtraDim end
	
	Neural network layer that adds an extra dimension to the 3D input-tensor.
	This is used to simuate 1D convolution with 2D convolution functions.
	"""
	struct AddExtraDim end
	(::AddExtraDim)(X) = reshape(X, size(X, 1), 1, size(X, 2), size(X, 3))
	Base.show(io::IO, ::AddExtraDim) = print(io, "AddExtraDim()")
	
	"""
		struct RemoveExtraDim end
	
	Neural network layer that removes the extra dimension added by 
	[`AddExtraDim`](@ref).
	"""
	struct RemoveExtraDim end
	(::RemoveExtraDim)(X) = reshape(X, size(X, 1), size(X, 3), size(X, 4))
	Base.show(io::IO, ::RemoveExtraDim) = print(io, "RemoveExtraDim()")

	"""
		mutable struct Dropout
			pdrop
			training 
		end
	
	Drop-out layers. If `training = true`, input dimensions are dropped 
	with probability `pdrop`.
	"""
	mutable struct Dropout pdrop; training end
	Dropout(pdrop) = Dropout(pdrop, true)
	(d::Dropout)(X) = dropout(X, d.pdrop, drop = d.training)
	Base.show(io::IO, d::Dropout) = print(io, "Dropout($(d.pdrop))")
	trainmode!(d::Dropout) = d.training = true
	testmode!(d::Dropout) = d.training = false
	
	"""
		struct Dense
			W
			b
			σ
		end
	
	Standard feed-forward layer with parameters `W` and `b` and activation
	function `σ`.
	"""
	struct Dense W; b; σ end 
	function (d::Dense)(X) 
		rX = reshape(X, size(X, 1), :)
		Y = d.σ.(d.W * rX .+ d.b)
		reshape(Y, :, size(X)[2:end]...)
	end
	Dense(indim::Int, outdim::Int, σ = identity) = Dense(
		param(outdim, indim, atype = Array{T}),
		param0(outdim, atype = Array{T}), # `param0` initialize the buffer to 0.
		σ
	)
	reallocate(d::Dense, atype) = Dense(
		Param(atype(value(d.W))),
		Param(atype(value(d.b))),
		d.σ
	)
	
	function Base.show(io::IO, d::Dense) 
		outdim, indim = size(d.W)
		print(io, "Dense($indim,  $outdim, $(d.σ))")
	end
	
	"""
		Conv(kernelsize, inchannels => outchannels; stride = (1, 1),
			 padding = (0, 0), dilation = (1, 1))
	
	2D convolution layer.
	"""
	struct Conv W; b; dims; stride; padding; dilation end
	function Conv(ksize, in_out; stride = (1, 1), padding = (0, 0),
				  dilation = (1, 1))
		W = param(ksize..., in_out.first, in_out.second, atype = Array{T})
		b = param0(1, 1, in_out.second, 1, atype = Array{T})
		Conv(W, b, in_out, stride, padding, dilation)
	end
	(c::Conv)(X) = conv4(c.W, X, padding = c.padding, stride = c.stride,
						 dilation = c.dilation) .+ c.b
	function reallocate(c::Conv, atype)
		Conv(
			Param(atype(value(c.W))),
			Param(atype(value(c.b))),
			c.dims, c.stride, c.padding, c.dilation
		)
	end
	function Base.show(io::IO, c::Conv) 
		s = (size(c.W, 1), size(c.W, 2))
		print(io, "Conv($s, $(c.dims), stride = $(c.stride), padding = $(c.padding), dilation = $(c.dilation))")
	end
	
	"""
		mutable struct BatchNorm
			σ
			training 
			...
		end
	
	Batch normalization layer.
	"""
	mutable struct BatchNorm
		moments
		params
		σ
		training
	end
	BatchNorm(dim, σ = identity) = BatchNorm(bnmoments(), Param(bnparams(T, dim)),
											 σ, true)
	(bn::BatchNorm)(X) = bn.σ.(batchnorm(X, bn.moments, bn.params; 
										 training = bn.training))
	trainmode!(bn::BatchNorm) = bn.training = true
	testmode!(bn::BatchNorm) = bn.training = false
	
	function reallocate(bn::BatchNorm, atype)
		mean, var = bn.moments.mean, bn.moments.var
		BatchNorm(
			bnmoments(;
				momentum = bn.moments.momentum,
				mean = isnothing(mean) ? nothing : atype(mean),
				var = isnothing(var) ? nothing : atype(var),
				meaninit = bn.moments.meaninit,
				varinit = bn.moments.varinit
			),
			Param(atype(value(bn.params))),
			bn.σ, bn.training
		)
	end
	
	Base.show(io::IO, bn::BatchNorm) =
		print(io, "BatchNorm($(length(bn.params) ÷ 2), $(bn.σ))")
	
	"""
		struct Chain 
			layers
		end
	
	Build a sequence of transformations.
	"""
	struct Chain
		layers
		Chain(layers...) = new(layers)
	end
	function (c::Chain)(X) 
		for layer in c.layers
			X = layer(X)
		end
		X
	end
	trainmode!(c::Chain) = trainmode!.(c.layers)
	testmode!(c::Chain) = testmode!.(c.layers)
	reallocate(c::Chain, atype) = Chain(reallocate.(c.layers, atype)...)
	
	Base.show(io::IO, c::Chain) = print(io, "Chain($(c.layers))")
	function Base.show(io::IO, ::MIME"text/plain", c::Chain)
		println(io, "Chain(")
		for layer in c.layers
			println(io, "    $(layer),")
		end
		print(io, ")")
	end
	
	md"""
	#### Neural Netword Layers
	"""
end

# ╔═╡ a82a8460-0e7e-4a0a-b22b-b1760796204e
md"""
### Building the model
"""

# ╔═╡ 3d95e008-866b-4670-a9a1-c948f6733464
function buildmodel(modelconfig, indim, outdim)
	dropout = modelconfig["dropout"]
	dilations = modelconfig["dilations"]
	hdims = modelconfig["hidden-dims"]
	ksizes = modelconfig["kernel-sizes"]
	strides = modelconfig["strides"]
	
	layers = []
	push!(layers, PermuteDims([2, 1, 3])) # Features dimensions as channels.
	push!(layers, AddExtraDim()) # To simulate 1D convolution.
	for i in 1:length(hdims)
		layer = Conv(
			(ksizes[i], 1), 
			indim => hdims[i],
			stride = (strides[i], 1),
			dilation = (dilations[i], 1),
			padding = (dilations[i]*(ksizes[i] - 1) ÷ 2, 0)
		)
		push!(layers, layer)
		push!(layers, BatchNorm(hdims[i], relu))
		push!(layers, Dropout(dropout))
		indim = hdims[i]
	end
	push!(layers, RemoveExtraDim())
	push!(layers, PermuteDims([2, 1, 3]))
	push!(layers, Dense(hdims[end], outdim))
		
	model = Chain(layers...)
end

# ╔═╡ 21585c4e-463a-4489-9c73-ac6d585f9c46
initmodel = if ispath(ckpt_path)
	load(ckpt_path, "model")
else
	buildmodel(config["model"], indim, outdim)
end

# ╔═╡ 1273528f-6d95-4b80-924c-e6a8eb7c7d34
init = if ispath(ckpt_path)
	load(ckpt_path)
else
	local trainconfig = config["training"]
	local model = use_gpu ? initmodel |> gpu : initmodel
	Dict(
		"model" => model,
		"optimizers" => [Adam(lr = trainconfig["optimizer"]["learning_rate"], 
	                          beta1 = trainconfig["optimizer"]["beta1"],
			                  beta2 = trainconfig["optimizer"]["beta2"])
		                 for p in params(initmodel)] ,
		"epoch" => 1,
		"bestloss" => Inf,
		"scheduler" => [Adam(lr = config["training"]["optimizer"]["learning_rate"], 
	             beta1 = config["training"]["optimizer"]["beta1"],
			     beta2 = config["training"]["optimizer"]["beta2"])
		    for p in params(initmodel)] 
	)
end

# ╔═╡ 1ad2d2e4-b23f-464c-b53a-47b7a8ea9649
md"""
The next function is needed to calculate the length of the sequences at the end of the neural network (after the downsampling).
"""

# ╔═╡ 0c081ba2-7b97-4aca-8608-36fc74114066
function getlengths(modelconfig, inlengths)
	dilations = modelconfig["dilations"]
	hdims = modelconfig["hidden-dims"]
	ksizes = modelconfig["kernel-sizes"]
	strides = modelconfig["strides"]
	
	seqlengths = inlengths
	for i in 1:length(dilations)
		ksize = ksizes[i]
		pad = dilations[i] * (ksize - 1) ÷ 2
		newlengths = seqlengths .+ 2*pad .- dilations[i]*(ksize - 1) .+ strides[i] .- 1 	
		seqlengths = newlengths .÷ strides[i]
	end
	seqlengths
end

# ╔═╡ f5e0931c-66a5-4bde-b7e0-d0391a234b23
md"""
## Training

### Data loader
"""

# ╔═╡ 5964d2d5-a61b-43a2-b37a-97e2e06331eb
function padutterance(utt, Nmax)
	D, N = size(utt)
	pad = similar(utt, D, Nmax - N)
	pad .= utt[:, end]
	hcat(utt, pad)
end

# ╔═╡ dcf4e7ed-15f1-46b9-9516-c082d7f482b2
function BatchLoader(h5data, alifsms, batchsize; shuffledata = false)
	uttids = sort(keys(h5data), by = k -> size(h5data[k], 2))
	bins = [uttids[i:min(i+batchsize-1, length(uttids))] 
		    for i in 1:batchsize:length(uttids)]
	
	# We only shuffle at the batch level.
	if shuffledata shuffle!(bins) end
	
	Channel() do ch
		for bin in bins
			# Load the features' utterance and build the batch.
			feas = [read(h5data[uttid]) for uttid in bin]
			seqlengths = [size(fea, 2) for fea in feas]
			Nmax = maximum(seqlengths)
			batch_feas = cat(padutterance.(feas, Nmax)..., dims = 3)
			
			# Load the numerator graphs.
			batch_alis = union([alifsms[uttid] for uttid in bin]...)
		
			put!(ch, (batch_feas, batch_alis, seqlengths))
		end
	end
end

# ╔═╡ bb446ac1-e770-41fb-8177-4d5b59d0bb8b
md"""
### Loss function

We define the the LF-MMI loss and its gradients.
"""

# ╔═╡ e6c62410-4439-4fc5-bf7c-a6f5abcdc082
function _lfmmi_loss(ϕ, numfsms, denfsms, seqlengths)
	t₀ = now()
	γ_num, ttl_num = pdfposteriors(numfsms, ϕ, seqlengths)
	γ_den, ttl_den = pdfposteriors(denfsms, ϕ, seqlengths)
	t₁ = now()
	etime = (t₁ - t₀).value / 1000 # Forward-Backward time for debugging.
	#@debug "forward-backward time = $etime"
	
	loss = -sum(ttl_num .- ttl_den)
	loss, γ_num, γ_den
end

# ╔═╡ 41f82983-0c37-4e88-a5e6-3a08ae51cf7b
_∇lfmmi_loss(γ_num, γ_den) = -(γ_num .- γ_den)

# ╔═╡ 24710344-74fe-4d6e-b003-c1bd49b529d8
@primitive1 _lfmmi_loss(ϕ, numf, denf, slen),dy,y (dy[1]*_∇lfmmi_loss(y[2:end]...))

# ╔═╡ 9bd4a7e7-4699-4306-9c61-1ad5ebad8e0a
lfmmi_loss(ϕ, numfsms, denfsms, seqlengths) = 
	_lfmmi_loss(ϕ, numfsms, denfsms, seqlengths)[1]

# ╔═╡ 5db4a224-2817-40f1-8ad1-bc282bd675a6
md"""
### Scheduler

We use a "plateau" scheduler, i.e. after `n` epochs without improvement, we decrease the learning rate.
"""

# ╔═╡ 7a0ed368-8d91-4e56-9d90-aeb741641666
mutable struct PlateauScheduler
	factor
	patience
	threshold
	best_loss
	nsteps
	min_lr
	
	PlateauScheduler(factor, patience, threshold, min_lr) =
		PlateauScheduler(factor, patience, threshold, Inf, 0, min_lr)
end

# ╔═╡ 85bf3355-3b20-417d-bde5-f9c05f2f0525
function update_scheduler!(s::PlateauScheduler, opts, loss)
	s.nsteps += 1
	if loss < s.best_loss * (1 - s.threshold)
		s.nsteps = 0
		s.best_loss = loss
	elseif s.nsteps > s.patience
		for opt in opts 
			opt.lr = max(opt.lr * s.factor, s.min_lr) 
		end
		@info "Decreasing the learning rate to $(opts[1].lr)."
		s.nsteps = 0
	end
	s
end

# ╔═╡ 784ea039-4944-4b48-aa23-1805b9266965
md"""
### Training
"""

# ╔═╡ 12f7ef0c-3b5a-4d24-8077-876d3718afa8
function train_epoch!(model, batchloader, denfsm, opts, use_gpu)
	trainmode!(model)
	θ = params(model)
	acc_loss = 0  
	acc_etime = 0
	N = 0  
	for (i, (batch_data, batch_nums, inlengths)) in enumerate(batchloader)
		GC.gc()
		CUDA.reclaim()

		seqlengths = getlengths(config["model"], inlengths)
		batch_dens = union(repeat([denfsm], size(batch_data, 3))...)

		if use_gpu 
			batch_data = CuArray(batch_data)
			batch_nums = MarkovModels.gpu(batch_nums)
		end

		L = @diff lfmmi_loss(model(batch_data), batch_nums, batch_dens, 
									seqlengths)
		update!(value.(θ), grad.([L], θ), opts)
		acc_loss += value(L)
		N += sum(seqlengths)

		L = nothing
	end
	acc_loss / N
end

# ╔═╡ 7222e9b4-8786-454c-9e2b-86c80bc27b29
function validate!(model, batchloader, denfsm, use_gpu)
	testmode!(model)
	acc_loss = 0  
	N = 0  
	for (i, (batch_data, batch_nums, inlengths)) in enumerate(batchloader)
		GC.gc()
		CUDA.reclaim()

		seqlengths = getlengths(config["model"], inlengths)
		batch_dens = union(repeat([denfsm], size(batch_data, 3))...)

		if use_gpu 
			batch_data = CuArray(batch_data)
			batch_nums = MarkovModels.gpu(batch_nums)
		end

		L = lfmmi_loss(model(batch_data), batch_nums, batch_dens, seqlengths)
		acc_loss += value(L) / sum(seqlengths)
		N += 1
	end
	acc_loss / N
end

# ╔═╡ 93e79378-fb11-4b7a-8571-7e61cd35c0ca
md"""
We check if there is a checkpoint. If not, we create a new model.
"""

# ╔═╡ 17aa86f4-7de5-41db-b104-07acfd9d5b88
if ispath(ckpt_path)
	
else
	scheduler = PlateauScheduler(
		config["training"]["scheduler"]["factor"],
		config["training"]["scheduler"]["patience"],
		config["training"]["scheduler"]["threshold"],
		Inf,
		0,
		config["training"]["scheduler"]["min_learning_rate"],
	)
	opts = [Adam(lr = config["training"]["optimizer"]["learning_rate"], 
	             beta1 = config["training"]["optimizer"]["beta1"],
			     beta2 = config["training"]["optimizer"]["beta2"])
		    for p in params(initmodel)] 
	startepoch = 1
	bestloss = Inf
end

# ╔═╡ 3041378b-a4f7-48d6-a4f2-3f19020104e0
function train!(model, trainconfig, trainfea, devfea, den_fsm, use_gpu, logfile)
	epochs = trainconfig["epochs"]
	curriculum = trainconfig["curriculum"]
	mbsize = trainconfig["minibatch_size"]
	ckpt_path = joinpath(traindir, trainconfig["checkpoint_file"])
	best_path = joinpath(traindir, trainconfig["best_file"])
	
	logger = ConsoleLogger(logfile)
	with_logger(logger) do 
		for epoch in startepoch:epochs
			t₀ = now()

			shuffledata = epoch > curriculum ? true : false
			train_loss = dev_loss = 0

			jldopen(train_numfsms_file, "r") do train_numfsms
				h5open(train_feafile, "r") do trainfea
					trainbl = BatchLoader(trainfea, train_numfsms, mbsize; 
										  shuffledata)
					train_loss = train_epoch!(model, trainbl, den_fsm, opts, use_gpu)
				end
			end

			jldopen(dev_numfsms_file, "r") do dev_numfsms
				h5open(dev_feafile, "r") do devfea
					devbl = BatchLoader(devfea, dev_numfsms, mbsize)
					dev_loss = validate!(model, devbl, den_fsm, use_gpu)
				end
			end
			
			update_scheduler!(scheduler, opts, dev_loss)
			
			checkpoint = Dict(
				"model" => model |> cpu,
				"optimizers" => opts,
				"epoch" => epoch,
				"bestloss" => bestloss,
				"scheduler" => scheduler
			)
			save(ckpt_path, checkpoint)
			if dev_loss < best_loss save(best_path, checkpoint) end
				

			t₁ = now()
			@info "epoch=$epoch/$epochs train_loss=$train_loss dev_loss=$dev_loss epoch_time=$((t₁ - t₀).value / 1000)"
			
			flush(logfile)
		end
	end
end

# ╔═╡ b0f22655-95cb-4e81-9c88-4af7f86e5d9e
const model = use_gpu ? initmodel |> gpu : initmodel |> cpu;

# ╔═╡ 6a5032b3-5461-41be-89e8-c62e0d1c247a
open(joinpath(traindir, "log.txt"), "w") do logfile 
	train_model!(model, config["training"], use_gpu, logfile)
end

# ╔═╡ 7c4b2d61-be7e-4cb0-b1dd-c34839b6b504
finalfile = joinpath(traindir, "final.jld2")

# ╔═╡ 15406cdc-d3dc-49fb-a315-f8b9d95685c6
finalmodel = reallocate(model, Array{T})

# ╔═╡ 7c53ac5f-0746-4588-b3d7-443b4ca75a9a
JLD2.save(finalfile, model = finalmodel)

# ╔═╡ 1826431b-650c-47ab-b391-ca6e1b28d193
JLD2.save(finalfile, Dict("model" => finalmodel))

# ╔═╡ de767fae-9b5d-43b6-ae2c-163f77704b61
md"""
## Test

We dump the output of the neural network for the test set.
"""

# ╔═╡ c045d97c-4e2c-4460-9588-9c9f8a2cc28a
md"""
### Data loader

Because we don't have alignment graphs for the test data, we need a special data loader.
"""

# ╔═╡ 1de913ef-d987-42ce-858a-05b3213da1a2
function DumpBatchLoader(h5data, batchsize)
	uttids = sort(keys(h5data), by = k -> size(h5data[k], 2))
	bins = [uttids[i:min(i+batchsize-1, length(uttids))] 
		    for i in 1:batchsize:length(uttids)]
	
	Channel() do ch
		for bin in bins
			# Load the features' utterance and build the batch.
			feas = [read(h5data[uttid]) for uttid in bin]
			seqlengths = [size(fea, 2) for fea in feas]
			Nmax = maximum(seqlengths)
			batch_feas = cat(padutterance.(feas, Nmax)..., dims = 3)
				
			put!(ch, (batch_feas, seqlengths, bin))
		end
	end
end

# ╔═╡ 0ced844c-48d4-454b-9517-d9493ada5307
md"""
### Dump neural network output
"""

# ╔═╡ da7bb83f-a968-4a56-9288-5db22f233ad3
begin
	const outdir2 = joinpath(expdir, "output")
	mkpath(outdir2)
	const outfile = joinpath(expdir, "output", "posteriors.h5")
end

# ╔═╡ 6f95324a-5085-429f-9737-10f4a714a3d9
function dump_output(model, feafile, outfile, mbsize, use_gpu)
	testmode!(model)
	
	h5open(feafile, "r") do fin
		h5open(outfile, "w") do fout
			bl = DumpBatchLoader(fin, mbsize)
			for (batch_data, inlengths, uttids) in bl
				if use_gpu
					batch_data = CuArray(batch_data)
				end
				
				seqlengths = getlengths(config["model"], inlengths)
				Y = Array(model(batch_data))
				for i in 1:size(batch_data, 3)
					Yᵢ = Y[:, 1:seqlengths[i], i]
					fout[uttids[i]] = Yᵢ
				end
			end
		end
	end
end		

# ╔═╡ dd980725-1250-4f5a-b13e-938c08f40a5a
outfile

# ╔═╡ 18e0a68d-7b44-4beb-90d6-9c045900f30d
dump_output(model, dev_feafile, outfile, 
			config["training"]["minibatch_size"], use_gpu)

# ╔═╡ 13189c78-cc2d-42b0-bca0-b57154c4c179
dump_output(model, dev_feafile, outfile, 1, use_gpu)

# ╔═╡ c401d3ba-1bfe-47dc-8dab-72f289882d08
begin
	#local model = initmodel |> gpu
	testmode!(model)
	local fea = h5open(train_feafile, "r") do f
		read(f["lbi-118-47824-0026"])
	end
	local numfsm = jldopen(train_numfsms_file, "r") do f
		f["lbi-118-47824-0026"] |> MarkovModels.gpu
	end
	local X = reshape(fea, 40, :, 1)
	
	local Y = model(CuArray(X))
	numY, ttl = pdfposteriors(union(numfsm), Y, [size(Y, 2)])
	numY = Array(numY)
end

# ╔═╡ 2979a46b-9bf0-4b9a-b458-8ae3e442cc8b
heatmap(reshape(numY, size(numY, 1), :))

# ╔═╡ 5ad5bd2c-0569-4d4b-b91b-62e0b167a501
begin
	local den_fsm = load(joinpath(graphsdir, "denominator_fsm.jld2"))["fsm"]

	Dict(state.label => state.initweight for state in filter(isinit, states(den_fsm)))
end

# ╔═╡ dc0bb517-6cef-403e-a9c6-ce13bf544b1e
begin
	#local model = initmodel |> gpu
	testmode!(model)
	local fea = h5open(train_feafile, "r") do f
		read(f["lbi-118-47824-0026"])
	end
	
	local den_fsm = load(joinpath(graphsdir, "denominator_fsm.jld2"))["fsm"]
	local den_fsm = use_gpu ? den_fsm |> MarkovModels.gpu : den_fsm
	
	local X = reshape(fea, 40, :, 1)
	
	local Y = model(CuArray(X))
	denY, den_ttl = pdfposteriors(union(den_fsm), Y, [size(Y, 2)])
	local denY = Array(denY)
end

# ╔═╡ 0967a299-6f79-46fa-bdc4-f1128ef50a7e
heatmap(reshape(denY, size(denY, 1), :))

# ╔═╡ 5d45406c-fa60-46bc-860f-7c8e750e702c
heatmap(reshape(abs.(numY .- denY), size(denY, 1), :))

# ╔═╡ ad5697ce-6bdf-4285-beb3-8c4b33d8e07a
jldopen(train_numfsms_file, "r") do f
	f["lbi-118-47824-0026"]
end

# ╔═╡ 64ee663f-ae27-459c-8c15-47d60a96cc56
const pdfid_mapping = jldopen(joinpath(graphsdir, "pdfid_mapping.jld2"), "r") do f
	f["pdfid_mapping"]
end

# ╔═╡ dfba8777-e9e1-42a7-9d93-8b7dcda5d863
with_terminal() do 
	
	testmode!(model)
	fea = h5open(dev_feafile, "r") do f
		read(f["lbi-84-121550-0012"])
	end
	
	den_fsm = load(joinpath(graphsdir, "denominator_fsm.jld2"))["fsm"]
	den_fsm = convert(MatrixFSM{TropicalSemiring{Float32}}, den_fsm)
	
	X = reshape(fea, 40, :, 1)
	Y = reshape(Array(model(CuArray(X))), 82, :)
	println(size(Y))
	
	Y = maxstateposteriors(den_fsm, reshape(Y, 82, :))
	path = bestpath(den_fsm, Y)
	
	outseq = []
	for t in getindex(den_fsm.labels, path)
		if t[end] == 1 # Select a HMM state.
			push!(outseq, t[end-1])
		end
	end
	println(join(outseq, " "))
end

# ╔═╡ 38e69333-63f5-4266-8c7b-476445066712
begin
	local fea = h5open(test_feafile, "r") do f
		read(f["lbi-1272-135031-0000"])
	end
	local X = reshape(fea, 40, :, 1)
	testmode!(model)
	reshape(Array(model(CuArray(X))), 82, :)
end

# ╔═╡ 4790e79e-f554-4bb8-b972-b724ebc85780
h5open("exp2/mini_librispeech/output/posteriors.h5", "r") do f
	read(f["lbi-1272-135031-0000"])
end

# ╔═╡ 440d491d-2bb6-4d19-ba9a-b38a957db42a
with_terminal() do 
	ϕ = h5open("exp2/mini_librispeech/output/posteriors.h5", "r") do f
		read(f["lbi-1272-135031-0000"])
	end
	
	den_fsm = load(joinpath(graphsdir, "denominator_fsm.jld2"))["fsm"]
	den_fsm = convert(MatrixFSM{TropicalSemiring{Float32}}, den_fsm)
	
	Y = maxstateposteriors(den_fsm, ϕ)
	path = bestpath(den_fsm, Y)
	
	outseq = []
	for t in getindex(den_fsm.labels, path)
		if t[end] == 1 # Select a HMM state.
			push!(outseq, t[end-1])
		end
	end
	println(join(outseq, " "))
end

# ╔═╡ 1aa7a1d7-bc0c-442f-9c25-84348376e2a1
model.layers[4].moments

# ╔═╡ Cell order:
# ╟─4df289c2-b460-49a6-b1b3-dac6c557397d
# ╠═fc878252-e6cd-4663-ad45-e42cae318ffb
# ╟─2ac818e7-7c52-4dfb-9f79-519fbce9d651
# ╠═876cffca-c46b-4805-a00f-5ac30d0631fe
# ╟─c10dd5eb-7bbe-4427-8a04-a3219d69942f
# ╟─850ac54c-b2e8-4e8e-a67a-b83f8d4d5905
# ╠═ee11cbaf-73c0-45db-84f7-db9eab7b6005
# ╟─ad6986d7-71b3-49cc-92e4-dadf42953b19
# ╠═bbde1167-ac11-4c15-8374-daa3f807cf3f
# ╟─89a40466-0e49-41b4-882a-20c66ad840b7
# ╠═20d81979-4fd8-4b51-9b27-70f54fb2f886
# ╟─016eabc9-127c-4e12-98ab-8cd5004edfa0
# ╠═104fc433-def8-44b1-b309-13d7683e0b33
# ╟─8bf454e4-7707-4184-a9b3-13a9d072576a
# ╠═7dbf440a-8550-48fe-869d-850e9cd79656
# ╠═8ab11e2e-af68-43d1-a766-7095719133ed
# ╠═ca199db4-442f-4caf-a84e-1b4bf73e6112
# ╠═b9f56cb7-5cea-4143-b4a4-251306bef902
# ╠═71f17a4a-88d5-449a-b0d0-fdee2956c96c
# ╟─82a88680-a834-49f2-b0ef-e6dbbc645131
# ╠═3d26b72a-65ba-4e6f-9e24-041007b5c413
# ╠═270cb75f-62ad-4435-9932-3ff37d6db89f
# ╟─aa271670-9e13-4838-968c-e15b34e47123
# ╠═8936b2c7-68b6-4845-94b5-6512a1cb16a2
# ╠═6e8c6723-d6d9-4dbe-acd7-c0575b1f832e
# ╠═994a581f-f5fa-4c1c-b772-8bb6c8f5a0f5
# ╠═6f336813-1d30-4dc4-a298-87019e7b7f36
# ╠═b5aa09be-a254-4b3e-81ed-91fe31b61c0e
# ╟─d137c3f8-ceb5-4fbb-8e46-72bdb397d072
# ╠═f3e1352d-9b13-4378-9b5d-6ff7f65a4005
# ╟─69f0e6a3-e620-4b55-951f-d32488a48fc5
# ╠═110a9455-f3f4-49bf-9d26-f6ed94440076
# ╟─a2d14acd-fed2-42b9-85dd-77c5e04b982d
# ╠═0b7d14dc-d580-4e5d-96c7-fd185deed2ed
# ╠═21585c4e-463a-4489-9c73-ac6d585f9c46
# ╟─67fd7845-47ca-4b3a-9b78-68db1586ea9e
# ╟─c8dc633e-81a6-41ee-b61f-90b26dc04494
# ╠═3041378b-a4f7-48d6-a4f2-3f19020104e0
# ╠═357e7c9d-9537-46f3-9153-b0bbccfd93b8
# ╠═1273528f-6d95-4b80-924c-e6a8eb7c7d34
# ╠═8b9b85ef-2419-4581-acca-31c033405587
# ╟─b95c2872-2dd4-4de5-ad81-c7b818ae21e0
# ╟─446c8edb-e129-4f2f-a676-1fcf502e24d6
# ╠═007c7b74-59f4-4154-9679-46169624aeea
# ╟─a82a8460-0e7e-4a0a-b22b-b1760796204e
# ╠═3d95e008-866b-4670-a9a1-c948f6733464
# ╟─1ad2d2e4-b23f-464c-b53a-47b7a8ea9649
# ╠═0c081ba2-7b97-4aca-8608-36fc74114066
# ╟─f5e0931c-66a5-4bde-b7e0-d0391a234b23
# ╠═5964d2d5-a61b-43a2-b37a-97e2e06331eb
# ╠═dcf4e7ed-15f1-46b9-9516-c082d7f482b2
# ╟─bb446ac1-e770-41fb-8177-4d5b59d0bb8b
# ╠═e6c62410-4439-4fc5-bf7c-a6f5abcdc082
# ╠═41f82983-0c37-4e88-a5e6-3a08ae51cf7b
# ╠═24710344-74fe-4d6e-b003-c1bd49b529d8
# ╠═9bd4a7e7-4699-4306-9c61-1ad5ebad8e0a
# ╟─5db4a224-2817-40f1-8ad1-bc282bd675a6
# ╠═7a0ed368-8d91-4e56-9d90-aeb741641666
# ╠═85bf3355-3b20-417d-bde5-f9c05f2f0525
# ╟─784ea039-4944-4b48-aa23-1805b9266965
# ╠═12f7ef0c-3b5a-4d24-8077-876d3718afa8
# ╠═7222e9b4-8786-454c-9e2b-86c80bc27b29
# ╟─93e79378-fb11-4b7a-8571-7e61cd35c0ca
# ╠═17aa86f4-7de5-41db-b104-07acfd9d5b88
# ╠═b0f22655-95cb-4e81-9c88-4af7f86e5d9e
# ╠═6a5032b3-5461-41be-89e8-c62e0d1c247a
# ╠═7c4b2d61-be7e-4cb0-b1dd-c34839b6b504
# ╠═15406cdc-d3dc-49fb-a315-f8b9d95685c6
# ╠═7c53ac5f-0746-4588-b3d7-443b4ca75a9a
# ╠═1826431b-650c-47ab-b391-ca6e1b28d193
# ╟─de767fae-9b5d-43b6-ae2c-163f77704b61
# ╟─c045d97c-4e2c-4460-9588-9c9f8a2cc28a
# ╠═1de913ef-d987-42ce-858a-05b3213da1a2
# ╟─0ced844c-48d4-454b-9517-d9493ada5307
# ╠═da7bb83f-a968-4a56-9288-5db22f233ad3
# ╠═6f95324a-5085-429f-9737-10f4a714a3d9
# ╠═dd980725-1250-4f5a-b13e-938c08f40a5a
# ╠═18e0a68d-7b44-4beb-90d6-9c045900f30d
# ╠═13189c78-cc2d-42b0-bca0-b57154c4c179
# ╠═6112bcf7-f4fb-428e-acba-aa64a2ddd555
# ╠═c401d3ba-1bfe-47dc-8dab-72f289882d08
# ╠═2979a46b-9bf0-4b9a-b458-8ae3e442cc8b
# ╠═5ad5bd2c-0569-4d4b-b91b-62e0b167a501
# ╠═dc0bb517-6cef-403e-a9c6-ce13bf544b1e
# ╠═0967a299-6f79-46fa-bdc4-f1128ef50a7e
# ╠═5d45406c-fa60-46bc-860f-7c8e750e702c
# ╠═ad5697ce-6bdf-4285-beb3-8c4b33d8e07a
# ╠═64ee663f-ae27-459c-8c15-47d60a96cc56
# ╠═dfba8777-e9e1-42a7-9d93-8b7dcda5d863
# ╠═38e69333-63f5-4266-8c7b-476445066712
# ╠═4790e79e-f554-4bb8-b972-b724ebc85780
# ╠═440d491d-2bb6-4d19-ba9a-b38a957db42a
# ╠═1aa7a1d7-bc0c-442f-9c25-84348376e2a1
