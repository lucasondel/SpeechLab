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
	# Lattice-Free MMI part II: training the model
	*[Lucas Ondel](https://lucasondel.github.io/index), September 2021*


	This notebook is the second step of the LF-MMI recipe. It implements the creation and the training of a *Time-Delay Neural Network* (TDNN) with the *Lattice-Free Maximum Mutual Information* (LF-MMI) objective function.

	Once the model is trained, the output of the neural network on the test set is dumped to an archive.
	"""
end

# ╔═╡ ee11cbaf-73c0-45db-84f7-db9eab7b6005
begin
	using AutoGrad
	using CUDA
	using Dates
	using JLD2
	using Knet
	using Logging
	using MarkovModels
	using HDF5
	using PlutoUI
	using Random
	using ShortCodes
	using TOML
end

# ╔═╡ fc878252-e6cd-4663-ad45-e42cae318ffb
TableOfContents()

# ╔═╡ 2ac818e7-7c52-4dfb-9f79-519fbce9d651
md"""
## Setup

Depending on your setup, select the CUDA version you want to use. You can also deactivate this cell and let the julia environment decide for itself.
"""

# ╔═╡ 876cffca-c46b-4805-a00f-5ac30d0631fe
ENV["JULIA_CUDA_VERSION"] = "11.3"

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
    for the neural-network/automatic differentiation backend with little modifications
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

# ╔═╡ 016eabc9-127c-4e12-98ab-8cd5004edfa0
md"""
We load the configuration file of the experiment. By default, we look for the file  `config.toml` in the root directory (i.e. the directory containing this notebook). Alternatively, when calling this notebook as a julia script, you can specify another configuration file by setting the environment variable `SPEECHLAB_LFMMI_CONFIG=/path/to/file`.
"""

# ╔═╡ 20d81979-4fd8-4b51-9b27-70f54fb2f886
rootdir = @__DIR__

# ╔═╡ 8ef896f4-c786-49f2-967b-72db140c933d
configfile = get(ENV, "SPEECHLAB_LFMMI_CONFIG", joinpath(rootdir, "config.toml"))

# ╔═╡ 104fc433-def8-44b1-b309-13d7683e0b33
begin
	@info "Reading configuration from $configfile."
	config = TOML.parsefile(configfile)
end

# ╔═╡ 1e09402a-a8b9-4e40-8bc2-2c0c153333c8
md"""
We use single precision.
"""

# ╔═╡ 65590fe8-f0e0-497f-bd8b-e128e23fd60b
const T = Float32

# ╔═╡ 8bf454e4-7707-4184-a9b3-13a9d072576a
md"""
## Input

Training with LF-MMI requires the numerator and denominator graphs (prepared by the previous step of the recipe) and the features for the training and development set.

The graphs should be organized as:
```
<rootdir>/
+-- <graphs.dir>/
|   +-- <dataset.name>/
|   |   +-- denominator_fsm.jld2
|   |   +-- dev_alignments_fsms.jld2
|   |   +-- pdfid_mapping.jld2
|   |   +-- train_alignments_fsms.jld2
```

As for the features, they should be in [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) archive organized as follows:
```
<features.dir>/
+-- <dataset.name>/
|   +-- train/
|   |   +-- <features.name>.h5
|   +-- dev/
|   |   +-- <features.name>.h5
|   +-- test/
|   |   +-- <features.name>.h5
```
where `<features.dir>`, `<features.name>` and `<dataset.name>` are taken from the configuration file.
!!! note
	The features directory path should be absolute and not relative to the root directory.
"""

# ╔═╡ 8ab11e2e-af68-43d1-a766-7095719133ed
graphsdir = joinpath(rootdir, config["graphs"]["dir"], config["dataset"]["name"])

# ╔═╡ 8936b2c7-68b6-4845-94b5-6512a1cb16a2
feadir = joinpath(config["features"]["dir"], config["dataset"]["name"])

# ╔═╡ 0b77b0f3-451f-41a9-a39a-da0525268ec5
feaname = config["features"]["name"]

# ╔═╡ 252656b1-0182-4616-9909-a7bc535244cf
featrain = joinpath(feadir, "train", feaname * ".h5")

# ╔═╡ 208c0a95-43ca-420c-b349-ac12da9631e4
feadev = joinpath(feadir, "dev", feaname * ".h5")

# ╔═╡ 907ad799-2f95-4b68-9033-7cc1d8853f00
featest = joinpath(feadir, "test", feaname * ".h5")

# ╔═╡ 270cb75f-62ad-4435-9932-3ff37d6db89f
use_gpu = config["training"]["use_gpu"]

# ╔═╡ aa271670-9e13-4838-968c-e15b34e47123
md"""
## Output

The notebook output the trained model and the output of the last layer on the train data.

```
<rootdir>/
+-- <experiment.dir>/
|   +-- <dataset.name>/
|   |   +-- <training.dir>/
|   |   |   +-- checkpoint.jld2
|   |   |   +-- best.jld2
|   |   +-- <output.dir>/
|   |   |   +-- test.h5
```
The keys `<experiment.dir>`, `<dataset.name>, `<training.dir>` and `<output.dir>` are taken from configuration file.

!!! note
	`best.jld2` is the model with the highest loss function on the development set whereas `checkpoint.jld2` is the model at the last epoch..

"""

# ╔═╡ eaa248f5-b8f3-4e3a-93a3-d4a275929878
expdir = joinpath(config["experiment"]["dir"], config["dataset"]["name"])

# ╔═╡ 6e8c6723-d6d9-4dbe-acd7-c0575b1f832e
traindir = joinpath(expdir, config["training"]["dir"])

# ╔═╡ d8de3103-bbe8-4858-8f97-ecebc3dd9103
outdir = joinpath(expdir, config["output"]["dir"])

# ╔═╡ 71f17a4a-88d5-449a-b0d0-fdee2956c96c
mkpath.([traindir, outdir])

# ╔═╡ d137c3f8-ceb5-4fbb-8e46-72bdb397d072
md"""
## Model creation

We use a TDNN model with ReLU activations. The input dimension corresponds to the input features dimension.
"""

# ╔═╡ f3e1352d-9b13-4378-9b5d-6ff7f65a4005
indim = h5open(featrain, "r") do f
	uttid, _ = iterate(keys(f))
	dim = size(read(f[uttid]), 1)
	@info "Neural network input dimension: $dim"
	dim
end

# ╔═╡ 69f0e6a3-e620-4b55-951f-d32488a48fc5
md"""
The output dimension corresponds to the number of pdf-ids.
"""

# ╔═╡ 110a9455-f3f4-49bf-9d26-f6ed94440076
outdim = jldopen(joinpath(graphsdir, "pdfid_mapping.jld2"), "r") do f
	dim = maximum( t -> t[2], f["pdfid_mapping"])
	@info "Neural network output dimension: $dim"
	dim
end

# ╔═╡ a2d14acd-fed2-42b9-85dd-77c5e04b982d
md"""
Now we build the model.
"""

# ╔═╡ 0b7d14dc-d580-4e5d-96c7-fd185deed2ed
ckpt_path = joinpath(traindir, "checkpoint.jld2")

# ╔═╡ 67fd7845-47ca-4b3a-9b78-68db1586ea9e
md"""
## Training
"""

# ╔═╡ e15b3f0f-6ec9-4bdf-9998-7b6359f6724f
md"""
First we load the phonotactic language model, i.e. the *denominator graph* of the MMI objective.
"""

# ╔═╡ 357e7c9d-9537-46f3-9153-b0bbccfd93b8
# Don't remove the end ';' as this object is too big to be displayed
den_fsm = load(joinpath(graphsdir, "denominator_fsm.jld2"), "fsm");

# ╔═╡ 19f0f3e2-dbce-4397-a9a2-12d6d0e9ea42
md"""
Now, we initialize the "training state". In our case, this comprises:
  * the parameters' optimizer
  * the best loss (on the validation set) achieved so far
  * the current epoch
  * the scheduler of the learning rate.
"""

# ╔═╡ c8dc633e-81a6-41ee-b61f-90b26dc04494
md"""
We define here the main function of the training. The input arguments are:

|Argument       | Description                                           |
|:--------------|:------------------------------------------------------|
| `model`       | Model to train                                        |
| `trainconfig` | Training part of the configuration of the experiment  |
| `trainstate`  | Current state of the training 						|
| `trainfea`    | Features archive for the train set                    |
| `trainnums`   | Numerator graphs for the train set                    |
| `devfea`      | Features archive for the train set                    |
| `devnums`     | Numerator graphs for the train set                    |
| `den_fsm`     | Phonotactic language model FSM for the denominator    |
| `use_gpu`     | Whether or not to use a GPU                           |
| `traindir`    | Output directory for the training 	                |
"""

# ╔═╡ 35aeec87-3267-4d07-8552-70d2c5da8d88
md"""
Let's train our model !
"""

# ╔═╡ 04c6afc4-c4cc-4df3-9eae-6b8e89d794f5
md"""
## Testing

Now the neural-network is trained and can be connected with a speech decoder.
As an example, we feed the test data to our model and store it in an HDF5 archive.
Because the HDF5 format can be read from most programming languages, you can probably use it with your favorite decoder 😉!

!!! note
	If you want to use a model already trained and stored on disk, you can simply load
    it by using:
	```julia
	load("/path/to/model.jld2", "model")
	```
"""

# ╔═╡ b66d91c2-9178-4074-bd20-e70d6475a0d5
md"""
Main function to dump the output of the network.
"""

# ╔═╡ 19166346-942f-4956-a83e-82ede6dfd2fa
md"""
Write the output to `<expdir>/output/test.h5`.
"""

# ╔═╡ d3ef660a-981a-4329-8565-9b1eeb393760
md"""
!!! tip
	With [python](https://www.python.org/), You can easily convert the HDF5 archive
	to a [Kaldi](https://github.com/kaldi-asr/kaldi) archive using
	[kaldi_io](https://github.com/vesis84/kaldi-io-for-python),
	[h5py](https://github.com/h5py/h5py) and
	[numpy](https://github.com/numpy/numpy). Here is an example:
	```python
	import h5py
	import kaldi_io
	import numpy as np

	h5archive = '<expdir>/output/test.h5'
	karchive = '<expdir>/output/test.ark'
	with h5py.File(h5archive) as fin:
		with open(karchive, 'wb') as fout:
			kaldi_io.write_mat(fout, np.array(fin[key]), key=key)
	```

"""

# ╔═╡ e881dd00-a629-4b72-8849-2615554053da
md"""
## References

Some references about TDNN and LF-MMI:
"""

# ╔═╡ 176d281d-d901-4fc5-ac44-2a1861793b50
[
	DOI("10.1109/29.21701"),
	DOI("10.21437/INTERSPEECH.2016-595"),
	DOI("10.21437/Interspeech.2018-1423"),
	DOI("10.21437/Interspeech.2020-3053")

]

# ╔═╡ f755a17b-6873-477d-9551-204b1ec858d9
md"""
## Utilities

Here is the implementations of all the utility functions we use in this notebook.
"""

# ╔═╡ 7df9cc64-79bb-45b5-b79b-98affd7d9d24
function update_scheduler!(s, opts, loss)
	s[:nsteps] += 1
	if loss < s[:best_loss] * (1 - s[:threshold])
		s[:nsteps] = 0
		s[:best_loss] = loss
	elseif s[:nsteps] > s[:patience]
		for opt in opts
			opt.lr = max(opt.lr * s[:factor], s[:min_lr])
		end
		@info "Decreasing the learning rate to $(opts[1].lr)."
		s[:nsteps] = 0
	end
	s
end

# ╔═╡ 154f2532-4e84-4a61-8860-5525be685243
md"""
### Neural Network API

Since KNet provides a rather low-level API to build neural network,
we define a few utilities that will make things easier for people
accustomed to pytorch/tensorflow. Unfold this cell if you want to
"""

# ╔═╡ f94e8e95-3ba7-437f-b3d1-ec4f2317a1fa
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
	cpu(obj) = reallocate(obj, Array{T})

	function reallocate(opt::Adam, atype)
        fstm = isnothing(opt.fstm) ? nothing : atype(opt.fstm)
        scndm = isnothing(opt.scndm) ? nothing : atype(opt.scndm)
		Adam(opt.lr, opt.beta1, opt.beta2, opt.eps, opt.t, opt.gclip,
			 fstm, scndm)
	end

	"""
		struct PermuteDims
			perm
		end

	Neural network layer that permutes the dimension of the input layer.
	"""
	struct PermuteDims perm end
	(p::PermuteDims)(X) = permutedims(X, p.perm)
	getstate(p::PermuteDims) = Dict()
	setstate!(p::PermuteDims, s) = nothing
	Base.show(io::IO, p::PermuteDims) = print(io, "PermuteDims($(p.perm))")

	"""
		struct AddExtraDim end

	Neural network layer that adds an extra dimension to the 3D input-tensor.
	This is used to simuate 1D convolution with 2D convolution functions.
	"""
	struct AddExtraDim end
	(::AddExtraDim)(X) = reshape(X, size(X, 1), 1, size(X, 2), size(X, 3))
	getstate(::AddExtraDim) = Dict()
	setstate!(::AddExtraDim, s) = nothing
	Base.show(io::IO, ::AddExtraDim) = print(io, "AddExtraDim()")

	"""
		struct RemoveExtraDim end

	Neural network layer that removes the extra dimension added by
	[`AddExtraDim`](@ref).
	"""
	struct RemoveExtraDim end
	(::RemoveExtraDim)(X) = reshape(X, size(X, 1), size(X, 3), size(X, 4))
	getstate(::RemoveExtraDim) = Dict()
	setstate!(::RemoveExtraDim, s) = nothing
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
	getstate(d::Dropout) = Dict(:pdrop => d.pdrop, :training => d.training)
	setstate!(d::Dropout, s) = (d.pdrop = s[:pdrop]; d.training = s[:training])
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
	mutable struct Dense W; b; σ end
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

	getstate(d::Dense) = Dict(:W => d.W, :b => d.b, :σ => d.σ)
	setstate!(d::Dense, s) = (d.W = s[:W]; d.b = s[:b]; d.σ = s[:σ])

	function Base.show(io::IO, d::Dense)
		outdim, indim = size(d.W)
		print(io, "Dense($indim,  $outdim, $(d.σ))")
	end

	"""
		Conv(kernelsize, inchannels => outchannels; stride = (1, 1),
			 padding = (0, 0), dilation = (1, 1))

	2D convolution layer.
	"""
	mutable struct Conv W; b; dims; stride; padding; dilation end
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
	getstate(d::Conv) = Dict(:W => d.W, :b => d.b, :dims => d.dims,
							 :stride => d.stride, :padding => d.padding,
							 :dilation => d.dilation)
	setstate!(d::Conv, s) = (d.W = s[:W]; d.b = s[:b]; d.dims = s[:dims];
							 d.stride = s[:stride]; d.padding = s[:padding];
							 d.dilation = s[:dilation])
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

	getstate(d::BatchNorm) = Dict(:moments => d.moments, :params => d.params,
								  :σ => d.σ, :training => d.training)
	setstate!(d::BatchNorm, s) = (d.moments = s[:moments]; d.params = s[:params];
								  d.σ = s[:σ]; d.training = s[:training])

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

	getstate(c::Chain) = getstate.(c.layers)
	setstate!(c::Chain, s) = setstate!.(c.layers, s)

	Base.show(io::IO, c::Chain) = print(io, "Chain($(c.layers))")
	function Base.show(io::IO, ::MIME"text/plain", c::Chain)
		println(io, "Chain(")
		for layer in c.layers
			println(io, "    $(layer),")
		end
		print(io, ")")
	end
end

# ╔═╡ a82a8460-0e7e-4a0a-b22b-b1760796204e
md"""
### TDNN construction
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

# ╔═╡ ed5d482b-0504-4716-bfc0-3a635db50704
initmodel = buildmodel(config["model"], indim, outdim)

# ╔═╡ 21585c4e-463a-4489-9c73-ac6d585f9c46
if ispath(ckpt_path)
	modelstate = load(ckpt_path, "model")
	setstate!(initmodel, modelstate)
end;

# ╔═╡ 1273528f-6d95-4b80-924c-e6a8eb7c7d34
trainstate = if ispath(ckpt_path)
	ckpt = load(ckpt_path)

	@info "Found existing checkpoint at $(ckpt_path)"
	@info "Training will start at epoch $(load(ckpt_path, "epoch"))"

	Dict(
		"optimizers" => load(ckpt_path, "optimizers"),
		"epoch" => load(ckpt_path, "epoch"),
		"bestloss" => load(ckpt_path, "bestloss"),
		"scheduler" => load(ckpt_path, "scheduler"),
	)
else
	local trainconfig = config["training"]
	Dict(
		"optimizers" => [Adam(lr = trainconfig["optimizer"]["learning_rate"],
	                          beta1 = trainconfig["optimizer"]["beta1"],
			                  beta2 = trainconfig["optimizer"]["beta2"])
		                 for p in params(initmodel)] ,
		"epoch" => 1,
		"bestloss" => Inf,
		"scheduler" => Dict(
			:factor => config["training"]["scheduler"]["factor"],
			:patience => config["training"]["scheduler"]["patience"],
			:threshold => config["training"]["scheduler"]["threshold"],
			:best_loss => Inf, # best loss so far
			:nsteps => 0, # current number of steps without improvment
			:min_lr => config["training"]["scheduler"]["min_learning_rate"]
		)
	)
end;

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

# ╔═╡ 8af48863-6b5f-496b-9c8f-2543fb412b59
md"""
### TDNN training
"""


# ╔═╡ 12f7ef0c-3b5a-4d24-8077-876d3718afa8
function train_epoch!(model, batchloader, denfsm, opts, updatefreq, use_gpu)
	trainmode!(model)
	θ = params(model)
	acc_loss = 0
	acc_etime = 0
	N = 0
	gs = nothing
	for (i, (batch_data, batch_nums, inlengths)) in enumerate(batchloader)

		seqlengths = getlengths(config["model"], inlengths)
		batch_dens = union(repeat([denfsm], size(batch_data, 3))...)

		if use_gpu
			batch_data = CuArray(batch_data)
			batch_nums = MarkovModels.gpu(batch_nums)
		end

		GC.gc()
		CUDA.reclaim()

		t₀ = now()
		L = @diff begin
            Y = model(batch_data)
            lfmmi_loss(Y, batch_nums, batch_dens, seqlengths)
        end
		t₁ = now()

		@debug "[batch=$i] loss+backprop time = $((t₁ - t₀).value / 1000)"
		gs = isnothing(gs) ? grad.([L], θ) : gs .+ grad.([L], θ)

		if i % updatefreq == 0
			update!(value.(θ), gs, opts)
            gs = nothing
		end
		acc_loss += value(L)
		N += sum(seqlengths)

        if use_gpu CUDA.unsafe_free!(batch_data) end
        batch_nums = nothing
		L = nothing
	end

    # If the gradients of the last batches were not used, make a
    # final update.
    if ! isnothing(gs)
        update!(value.(θ), gs, opts)
    end

	acc_loss / N
end

# ╔═╡ f5e0931c-66a5-4bde-b7e0-d0391a234b23
md"""
### Loading data
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

# ╔═╡ 8ec6d305-7a49-47f1-a051-b2a1195e7555
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

# ╔═╡ bb446ac1-e770-41fb-8177-4d5b59d0bb8b
md"""
### Loss function

The LF-MMI loss and its gradients.
"""

# ╔═╡ e6c62410-4439-4fc5-bf7c-a6f5abcdc082
function _lfmmi_loss(ϕ, numfsms, denfsms, seqlengths)
	t₀ = now()
	γ_num, ttl_num = pdfposteriors(numfsms, ϕ, seqlengths)
	γ_den, ttl_den = pdfposteriors(denfsms, ϕ, seqlengths)
	t₁ = now()
	etime = (t₁ - t₀).value / 1000 # Forward-Backward time for debugging.
	@debug "forward-backward time = $etime"

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

# ╔═╡ 7222e9b4-8786-454c-9e2b-86c80bc27b29
function validate!(model, batchloader, denfsm, use_gpu)
	testmode!(model)
	acc_loss = 0
	N = 0
	for (i, (batch_data, batch_nums, inlengths)) in enumerate(batchloader)
		#GC.gc()
		#CUDA.reclaim()

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

# ╔═╡ 3041378b-a4f7-48d6-a4f2-3f19020104e0
function train_model!(model, trainconfig, trainstate, trainfea, trainnums,
					  devfea, devnums, den_fsm, use_gpu, traindir)
    total_t₀ = now()

	epochs = trainconfig["epochs"]
	curriculum = trainconfig["curriculum"]
	mbsize = trainconfig["minibatch_size"]
	ckpt_path = joinpath(traindir, "checkpoint.jld2")
	best_path = joinpath(traindir, "best.jld2")

	startepoch = trainstate["epoch"]
	opts = trainstate["optimizers"]
	scheduler = trainstate["scheduler"]
	bestloss = trainstate["bestloss"]
	bestmodel = nothing

	opts = use_gpu ? opts .|> gpu : opts .|> cpu

	for epoch in startepoch:epochs
		t₀ = now()

		shuffledata = epoch > curriculum ? true : false
		train_loss = dev_loss = 0

		jldopen(trainnums, "r") do train_numfsms
			h5open(trainfea, "r") do f
				trainbl = BatchLoader(f, train_numfsms, mbsize; shuffledata)
				train_loss = train_epoch!(model, trainbl, den_fsm, opts,
                                          trainconfig["update_freq"], use_gpu)
			end
		end

		jldopen(devnums, "r") do dev_numfsms
			h5open(devfea, "r") do f
				devbl = BatchLoader(f, dev_numfsms, mbsize)
				dev_loss = validate!(model, devbl, den_fsm, use_gpu)
			end
		end

		scheduler = update_scheduler!(scheduler, opts, dev_loss)

		checkpoint = Dict(
			"model" => getstate(model |> cpu),
			"optimizers" => opts .|> cpu,
			"epoch" => epoch+1,
			"bestloss" => bestloss,
			"scheduler" => scheduler
		)
		save(ckpt_path, checkpoint)
		if dev_loss < bestloss
			bestmodel = model |> cpu
			save(best_path, checkpoint)
		end


		t₁ = now()
		@info "epoch=$epoch/$epochs train_loss=$train_loss dev_loss=$dev_loss epoch_time=$((t₁ - t₀).value / 1000)"
	end
    total_t₁ = now()
    @debug "Total training time = $((total_t₁ - total_t₀).value / 1000) seconds."

	bestmodel
end

# ╔═╡ f6ecb7d9-9bcc-4076-94b8-7755a17f2a49
model = train_model!(
	use_gpu ? initmodel |> gpu : initmodel |> cpu,
	config["training"],
	trainstate,
	featrain,
	joinpath(graphsdir, "train_alignments_fsms.jld2"),
	feadev,
	joinpath(graphsdir, "dev_alignments_fsms.jld2"),
	use_gpu ? den_fsm |> MarkovModels.gpu : den_fsm,
	use_gpu,
	traindir
)

# ╔═╡ 988480f8-9666-4626-b404-ea0f69f96d4e
dump_output(
	use_gpu ? model |> gpu : model |> cpu,
	featest,
	joinpath(outdir, "test.h5"),
	config["training"]["minibatch_size"],
	use_gpu
)

# ╔═╡ Cell order:
# ╠═4df289c2-b460-49a6-b1b3-dac6c557397d
# ╠═fc878252-e6cd-4663-ad45-e42cae318ffb
# ╟─2ac818e7-7c52-4dfb-9f79-519fbce9d651
# ╠═876cffca-c46b-4805-a00f-5ac30d0631fe
# ╟─c10dd5eb-7bbe-4427-8a04-a3219d69942f
# ╟─850ac54c-b2e8-4e8e-a67a-b83f8d4d5905
# ╠═ee11cbaf-73c0-45db-84f7-db9eab7b6005
# ╟─ad6986d7-71b3-49cc-92e4-dadf42953b19
# ╠═bbde1167-ac11-4c15-8374-daa3f807cf3f
# ╟─016eabc9-127c-4e12-98ab-8cd5004edfa0
# ╠═20d81979-4fd8-4b51-9b27-70f54fb2f886
# ╠═8ef896f4-c786-49f2-967b-72db140c933d
# ╠═104fc433-def8-44b1-b309-13d7683e0b33
# ╟─1e09402a-a8b9-4e40-8bc2-2c0c153333c8
# ╠═65590fe8-f0e0-497f-bd8b-e128e23fd60b
# ╠═8bf454e4-7707-4184-a9b3-13a9d072576a
# ╠═8ab11e2e-af68-43d1-a766-7095719133ed
# ╠═8936b2c7-68b6-4845-94b5-6512a1cb16a2
# ╠═0b77b0f3-451f-41a9-a39a-da0525268ec5
# ╠═252656b1-0182-4616-9909-a7bc535244cf
# ╠═208c0a95-43ca-420c-b349-ac12da9631e4
# ╠═907ad799-2f95-4b68-9033-7cc1d8853f00
# ╠═270cb75f-62ad-4435-9932-3ff37d6db89f
# ╟─aa271670-9e13-4838-968c-e15b34e47123
# ╠═eaa248f5-b8f3-4e3a-93a3-d4a275929878
# ╠═6e8c6723-d6d9-4dbe-acd7-c0575b1f832e
# ╠═d8de3103-bbe8-4858-8f97-ecebc3dd9103
# ╠═71f17a4a-88d5-449a-b0d0-fdee2956c96c
# ╟─d137c3f8-ceb5-4fbb-8e46-72bdb397d072
# ╠═f3e1352d-9b13-4378-9b5d-6ff7f65a4005
# ╟─69f0e6a3-e620-4b55-951f-d32488a48fc5
# ╠═110a9455-f3f4-49bf-9d26-f6ed94440076
# ╟─a2d14acd-fed2-42b9-85dd-77c5e04b982d
# ╠═ed5d482b-0504-4716-bfc0-3a635db50704
# ╠═0b7d14dc-d580-4e5d-96c7-fd185deed2ed
# ╠═21585c4e-463a-4489-9c73-ac6d585f9c46
# ╟─67fd7845-47ca-4b3a-9b78-68db1586ea9e
# ╟─e15b3f0f-6ec9-4bdf-9998-7b6359f6724f
# ╠═357e7c9d-9537-46f3-9153-b0bbccfd93b8
# ╟─19f0f3e2-dbce-4397-a9a2-12d6d0e9ea42
# ╠═1273528f-6d95-4b80-924c-e6a8eb7c7d34
# ╟─c8dc633e-81a6-41ee-b61f-90b26dc04494
# ╠═3041378b-a4f7-48d6-a4f2-3f19020104e0
# ╟─35aeec87-3267-4d07-8552-70d2c5da8d88
# ╠═f6ecb7d9-9bcc-4076-94b8-7755a17f2a49
# ╟─04c6afc4-c4cc-4df3-9eae-6b8e89d794f5
# ╟─b66d91c2-9178-4074-bd20-e70d6475a0d5
# ╠═8ec6d305-7a49-47f1-a051-b2a1195e7555
# ╟─19166346-942f-4956-a83e-82ede6dfd2fa
# ╠═988480f8-9666-4626-b404-ea0f69f96d4e
# ╟─d3ef660a-981a-4329-8565-9b1eeb393760
# ╟─e881dd00-a629-4b72-8849-2615554053da
# ╟─176d281d-d901-4fc5-ac44-2a1861793b50
# ╟─f755a17b-6873-477d-9551-204b1ec858d9
# ╠═7df9cc64-79bb-45b5-b79b-98affd7d9d24
# ╟─154f2532-4e84-4a61-8860-5525be685243
# ╠═f94e8e95-3ba7-437f-b3d1-ec4f2317a1fa
# ╟─a82a8460-0e7e-4a0a-b22b-b1760796204e
# ╠═3d95e008-866b-4670-a9a1-c948f6733464
# ╟─1ad2d2e4-b23f-464c-b53a-47b7a8ea9649
# ╠═0c081ba2-7b97-4aca-8608-36fc74114066
# ╟─8af48863-6b5f-496b-9c8f-2543fb412b59
# ╠═12f7ef0c-3b5a-4d24-8077-876d3718afa8
# ╠═7222e9b4-8786-454c-9e2b-86c80bc27b29
# ╟─f5e0931c-66a5-4bde-b7e0-d0391a234b23
# ╠═5964d2d5-a61b-43a2-b37a-97e2e06331eb
# ╠═dcf4e7ed-15f1-46b9-9516-c082d7f482b2
# ╠═1de913ef-d987-42ce-858a-05b3213da1a2
# ╟─bb446ac1-e770-41fb-8177-4d5b59d0bb8b
# ╠═e6c62410-4439-4fc5-bf7c-a6f5abcdc082
# ╠═41f82983-0c37-4e88-a5e6-3a08ae51cf7b
# ╠═24710344-74fe-4d6e-b003-c1bd49b529d8
# ╠═9bd4a7e7-4699-4306-9c61-1ad5ebad8e0a
