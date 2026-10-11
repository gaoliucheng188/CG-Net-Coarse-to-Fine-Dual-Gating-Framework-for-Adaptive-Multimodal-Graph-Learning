#!/bin/bash

echo "========================================================"
echo "Extracting Case Studies (Task 3)"
echo "========================================================"

echo "1. MUTAG"
python analysis_scripts/extract_case_study.py --dataset mutag --ckpt-path ckpt/mutag_base_h128_o128_full_modal.pth.tar --outdir analysis_results/case_studies

echo "2. NLM-Chem"
python analysis_scripts/extract_case_study.py --dataset nlm_chem --ckpt-path ckpt/nlm_chem_base_h128_o128_full_modal.pth.tar --outdir analysis_results/case_studies

echo "3. PQARef"
python analysis_scripts/extract_case_study.py --dataset pqaref --ckpt-path ckpt/pqaref_base_h128_o128_full_modal.pth.tar --outdir analysis_results/case_studies

echo "Case Studies Extraction Completed."

