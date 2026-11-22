# One button:  make
#   creates .venv, installs everything, downloads the embedding model, runs every stage.
# Put GM's extract in data/ first (see data/README.md).

PYTHON  ?= python3
VENV    := .venv
PY      := $(VENV)/bin/python
BERTOPIC ?= 1
STAMP   := .make

.PHONY: all setup models run llm demo test clean clean-outputs help

all: run                                   ## setup + model + full pipeline

help:                                      ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

$(STAMP)/setup: requirements.txt requirements-bertopic.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@if [ "$(BERTOPIC)" = "1" ]; then $(PY) -m pip install -r requirements-bertopic.txt; fi
	$(PY) -m spacy download en_core_web_sm
	@mkdir -p $(STAMP) && touch $@

setup: $(STAMP)/setup                      ## create .venv and install dependencies

$(STAMP)/models: $(STAMP)/setup
	@if [ "$(BERTOPIC)" = "1" ]; then $(PY) tools/download_model.py; else echo "BERTOPIC=0: skipping model"; fi
	@touch $@

models: $(STAMP)/models                    ## download + verify the BERTopic embedding model

run: $(STAMP)/models                       ## run every stage enabled in config.yaml
	$(PY) run_pipeline.py

llm: $(STAMP)/models                       ## full pipeline plus TopicGPT/LLooM (needs .env; sends text)
	$(PY) run_pipeline.py --with-llm

demo: $(STAMP)/models                      ## full pipeline on a synthetic extract -> outputs_demo/
	$(PY) tools/synthetic_extract.py data/synthetic_extract.xlsx --rows 1500
	$(PY) run_pipeline.py --input data/synthetic_extract.xlsx --outputs outputs_demo

test: $(STAMP)/setup                       ## run the test suite (synthetic data, no network)
	$(PY) -m pytest -q tests

clean-outputs:                             ## delete generated outputs
	rm -rf outputs outputs_demo

clean: clean-outputs                       ## delete outputs, .venv and caches
	rm -rf $(VENV) $(STAMP) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
