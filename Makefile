SHELL := /bin/bash

REPO_DIR := automatic-on-policy-self-distillation
SYNC_SCRIPT := ./sync-to-slurm-gpu.sh

.PHONY: sync shell sync-shell g l sync-dir

sync:
	$(SYNC_SCRIPT)

g: SYNC_DIR := g/$(REPO_DIR)
g: sync-dir

l: SYNC_DIR := luke/$(REPO_DIR)
l: sync-dir

sync-dir:
	$(SYNC_SCRIPT) "$(SYNC_DIR)"

shell:
	ssh slurm-gpu

sync-shell: sync
	ssh slurm-gpu