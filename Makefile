SHELL := /bin/bash

.PHONY: sync shell sync-shell

sync:
	./sync-to-slurm-gpu.sh

shell:
	ssh slurm-gpu

sync-shell: sync
	ssh slurm-gpu
