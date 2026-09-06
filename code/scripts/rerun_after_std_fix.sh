#!/usr/bin/env bash
# Full pipeline rerun after the weight_based_attribution std-standardization fix
# (see NOTES.md "Design correction" section). Every result file that used
# weight_based_attribution was deleted; this regenerates all of them with the
# corrected design (std borrowed from the neuron's own layer, not target_layer).
#
# Run from the `code/` directory with a GPU available:
#   bash scripts/rerun_after_std_fix.sh
#
# Dependency order matters: flow1_analysis.py's output feeds
# baselines_comparison.py (--existing-results), and flow2_rq1_overlap.py's
# output feeds both steering_experiment.py and typological_proximity.py
# (--rq1-results). Everything within a stage is independent and safe to run
# in parallel across GPUs (edit --device below) if you don't want to wait
# for this script's sequential run.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyohyeongjang/.conda/envs/commonRegion/bin/python
mkdir -p results logs

echo "=== Stage 1: flow1_analysis.py (LAPE + attribution + patching, RQ1/2/4/5/6) ==="

$PY scripts/flow1_analysis.py --model pythia-70m \
    --n-examples-per-lang 1500 --n-neurons 150 --device cuda:0 \
    --out results/flow1_pythia70m_realscale.json 2>&1 | tee logs/flow1_pythia70m_analysis.log

$PY scripts/flow1_analysis.py --model pythia-410m \
    --mlsae-checkpoint checkpoints/mlsae_pythia410m.pt --mlsae-config checkpoints/mlsae_pythia410m.pt.config.json \
    --n-examples-per-lang 1500 --n-neurons 150 --device cuda:0 \
    --out results/flow1_pythia410m_realscale.json 2>&1 | tee logs/flow1_pythia410m_analysis.log

$PY scripts/flow1_analysis.py --model bloom-560m \
    --mlsae-checkpoint checkpoints/mlsae_bloom560m.pt --mlsae-config checkpoints/mlsae_bloom560m.pt.config.json \
    --n-examples-per-lang 1500 --n-neurons 150 --device cuda:0 \
    --out results/flow1_bloom560m_realscale.json 2>&1 | tee logs/flow1_bloom560m_analysis.log

$PY scripts/flow1_analysis.py --model bloom-1b7 \
    --mlsae-checkpoint checkpoints/mlsae_bloom1b7.pt --mlsae-config checkpoints/mlsae_bloom1b7.pt.config.json \
    --n-examples-per-lang 1500 --n-neurons 150 --device cuda:0 \
    --out results/flow1_bloom1b7_realscale.json 2>&1 | tee logs/flow1_bloom1b7_analysis.log

echo "=== Stage 2: flow2_rq1_overlap.py (SAE-LAPE / monolinguality overlap, BLOOM only) ==="

$PY scripts/flow2_rq1_overlap.py --model-key bloom-560m --model-name bigscience/bloom-560m \
    --mlsae-checkpoint checkpoints/mlsae_bloom560m.pt --mlsae-config checkpoints/mlsae_bloom560m.pt.config.json \
    --n-examples-per-lang 1500 --n-neurons 150 --device cuda:0 \
    --out results/flow2_bloom560m_rq1.json 2>&1 | tee logs/flow2_bloom560m_rq1.log

$PY scripts/flow2_rq1_overlap.py --model-key bloom-1b7 --model-name bigscience/bloom-1b7 \
    --mlsae-checkpoint checkpoints/mlsae_bloom1b7.pt --mlsae-config checkpoints/mlsae_bloom1b7.pt.config.json \
    --n-examples-per-lang 1500 --n-neurons 150 --device cuda:0 \
    --out results/flow2_bloom1b7_rq1.json 2>&1 | tee logs/flow2_bloom1b7_rq1.log

echo "=== Stage 3: baselines_comparison.py (correlational + GradSAE baselines) ==="

$PY scripts/baselines_comparison.py --model pythia-70m \
    --existing-results results/flow1_pythia70m_realscale.json --device cuda:0 \
    --out results/baselines_pythia70m.json 2>&1 | tee logs/baselines_pythia70m.log

$PY scripts/baselines_comparison.py --model pythia-410m \
    --mlsae-checkpoint checkpoints/mlsae_pythia410m.pt --mlsae-config checkpoints/mlsae_pythia410m.pt.config.json \
    --existing-results results/flow1_pythia410m_realscale.json --device cuda:0 \
    --out results/baselines_pythia410m.json 2>&1 | tee logs/baselines_pythia410m.log

$PY scripts/baselines_comparison.py --model bloom-560m \
    --mlsae-checkpoint checkpoints/mlsae_bloom560m.pt --mlsae-config checkpoints/mlsae_bloom560m.pt.config.json \
    --existing-results results/flow1_bloom560m_realscale.json --device cuda:0 \
    --out results/baselines_bloom560m.json 2>&1 | tee logs/baselines_bloom560m.log

$PY scripts/baselines_comparison.py --model bloom-1b7 \
    --mlsae-checkpoint checkpoints/mlsae_bloom1b7.pt --mlsae-config checkpoints/mlsae_bloom1b7.pt.config.json \
    --existing-results results/flow1_bloom1b7_realscale.json --device cuda:0 \
    --out results/baselines_bloom1b7.json 2>&1 | tee logs/baselines_bloom1b7.log

echo "=== Stage 4: steering_experiment.py (BLOOM only, RQ6 of flow_2.md) ==="

$PY scripts/steering_experiment.py --model-key bloom-560m --model-name bigscience/bloom-560m \
    --mlsae-checkpoint checkpoints/mlsae_bloom560m.pt --mlsae-config checkpoints/mlsae_bloom560m.pt.config.json \
    --rq1-results results/flow2_bloom560m_rq1.json --device cuda:0 \
    --out results/steering_bloom560m.json 2>&1 | tee logs/steering_bloom560m.log

$PY scripts/steering_experiment.py --model-key bloom-1b7 --model-name bigscience/bloom-1b7 \
    --mlsae-checkpoint checkpoints/mlsae_bloom1b7.pt --mlsae-config checkpoints/mlsae_bloom1b7.pt.config.json \
    --rq1-results results/flow2_bloom1b7_rq1.json --device cuda:0 \
    --out results/steering_bloom1b7.json 2>&1 | tee logs/steering_bloom1b7.log

echo "=== Stage 5: typological_proximity.py (BLOOM only, RQ4 of flow_2.md) ==="

$PY scripts/typological_proximity.py --rq1-results results/flow2_bloom560m_rq1.json \
    --out results/typological_bloom560m.json 2>&1 | tee logs/typological_bloom560m.log

$PY scripts/typological_proximity.py --rq1-results results/flow2_bloom1b7_rq1.json \
    --out results/typological_bloom1b7.json 2>&1 | tee logs/typological_bloom1b7.log

echo "=== All stages done. ==="
