# SPDX-License-Identifier: MIT

include Makefile.config

.PHONY: check_config features stats train

all: check_config features stats train

check_config:
ifndef DATASET
	$(error ERROR: the variable DATASET is not set)
endif

features:
	$(MAKE) -C $(ROOT)/recipes/features DATASET=$(DATASET)

stats: $(GMM_DIR)/stats.bson
train: $(GMM_DIR)/gmm.jld2

$(GMM_DIR)/stats.bson: $(SCRIPTS)/feastats.jl
	@echo -------------------------------------------------------------
	@echo Collecting statistics
	@echo -------------------------------------------------------------
	mkdir -p $(GMM_DIR)
	cat $(DATASETS)/$(DATASET)/uttids | head -n $(GMM_NUTTS) | \
		$(JULIA) $(SCRIPTS)/feastats.jl $(FEA_DIR) $(GMM_DIR)/stats.bson


$(GMM_DIR)/gmm.jld2: $(GMM_DIR)/stats.bson $(SCRIPTS)/gmm.jl $(CONFIGS)/gmm.toml
	@echo -------------------------------------------------------------
	@echo Training
	@echo -------------------------------------------------------------
	$(JULIA) $(SCRIPTS)/gmm.jl \
		$(GMM_TRAIN_OPTS) \
		$(CONFIGS)/gmm.toml \
		$(GMM_DIR)/stats.bson \
		$(DATASETS)/$(DATASET)/uttids \
		$(FEA_DIR) \
		$(GMM_DIR)/gmm.jld2


