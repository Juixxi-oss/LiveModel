#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p log/SnapRegressor



######################################################################################
nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type C --photo_type D --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_C_photoD.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type C --photo_type C --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_C_photoC.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type C --photo_type N --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_C_photoN.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type C --photo_type P --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_C_photoP.txt 2>&1 &


######################################################################################
nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type N --photo_type D --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_N_photoD.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type N --photo_type C --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_N_photoC.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type N --photo_type N --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_N_photoN.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type N --photo_type P --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_N_photoP.txt 2>&1 &


######################################################################################
nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type P --photo_type D --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_P_photoD.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type P --photo_type C --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_P_photoC.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type P --photo_type N --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_P_photoN.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type P --photo_type P --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_P_photoP.txt 2>&1 &


######################################################################################
nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type BCVA --photo_type D --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_BCVA_photoD.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type BCVA --photo_type C --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_BCVA_photoC.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type BCVA --photo_type N --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_BCVA_photoN.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type BCVA --photo_type P --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_BCVA_photoP.txt 2>&1 &


######################################################################################
nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type MTF --photo_type D --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_MTF_photoD.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type MTF --photo_type C --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_MTF_photoC.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type MTF --photo_type N --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_MTF_photoN.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type MTF --photo_type P --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_MTF_photoP.txt 2>&1 &


######################################################################################
nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type OSI --photo_type D --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_OSI_photoD.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type OSI --photo_type C --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_OSI_photoC.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type OSI --photo_type N --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_OSI_photoN.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type OSI --photo_type P --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_OSI_photoP.txt 2>&1 &


######################################################################################
nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type SR --photo_type D --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_SR_photoD.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type SR --photo_type C --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_SR_photoC.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:1 --grade_type SR --photo_type N --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_SR_photoN.txt 2>&1 &

nohup python -u main/SnapRegressor_main.py \
    --device cuda:3 --grade_type SR --photo_type P --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    > log/SnapRegressor/SnapRegressor_SR_photoP.txt 2>&1 &
