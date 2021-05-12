# SPDX-License-Identifier: MIT

include Makefile.config

.PHONY: features

all: check_config extractfeatures

check_config:
ifndef DATASET
	$(error ERROR: the variable DATASET is not set)
endif

extractfeatures: $(FEA_DIR)/.done

$(FEA_DIR)/.done: $(FEA_CONFIG)
	@echo -------------------------------------------------------------
	@echo Extracting features for the $(DATASET) set.
	@echo -------------------------------------------------------------
	mkdir -p $(FEA_DIR)
	$(JULIA) $(SCRIPTS)/features.jl \
		--parallel-njobs $(FEA_PARALLEL_NJOBS) \
		--parallel-args $(FEA_PARALLEL_ARGS) \
		$(FEA_CONFIG) \
		$(DATASETS)/$(DATASET)/wav.scp \
		$(FEA_DIR)
	date > $(FEA_DIR)/.done
