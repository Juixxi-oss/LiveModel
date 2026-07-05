cd /data_ssd/juixxi/LiveModel
export PYTHONPATH=/data_ssd/juixxi/LiveModel
source activate env_1
clear



nohup python -u main/ViewFusionRegressor_main.py \
    --device cuda:4 --grade_type C --num_folds 10 --start_fold 0 --num_epochs 50 --batch_size 64 \
    --lr 1e-4 --seed 42 > log/ViewFusionRegressor/ViewFusionRegressor_C.txt 2>&1 &

nohup python -u main/ViewFusionRegressor_main.py \
    --device cuda:4 --grade_type N --num_folds 10 --start_fold 0 --num_epochs 50 --batch_size 64 \
    --lr 1e-4 --seed 42 > log/ViewFusionRegressor/ViewFusionRegressor_N.txt 2>&1 &

nohup python -u main/ViewFusionRegressor_main.py \
    --device cuda:4 --grade_type P --num_folds 10 --start_fold 0 --num_epochs 50 --batch_size 64 \
    --lr 1e-4 --seed 42 > log/ViewFusionRegressor/ViewFusionRegressor_P.txt 2>&1 &

nohup python -u main/ViewFusionRegressor_main.py \
    --device cuda:4 --grade_type BCVA --num_folds 10 --start_fold 0 --num_epochs 50 --batch_size 64 \
    --lr 1e-4 --seed 42 > log/ViewFusionRegressor/ViewFusionRegressor_BCVA.txt 2>&1 &

nohup python -u main/ViewFusionRegressor_main.py \
    --device cuda:4 --grade_type MTF --num_folds 10 --start_fold 0 --num_epochs 50 --batch_size 64 \
    --lr 1e-4 --seed 42 > log/ViewFusionRegressor/ViewFusionRegressor_MTF.txt 2>&1 &

nohup python -u main/ViewFusionRegressor_main.py \
    --device cuda:4 --grade_type OSI --num_folds 10 --start_fold 0 --num_epochs 50 --batch_size 64 \
    --lr 1e-4 --seed 42 > log/ViewFusionRegressor/ViewFusionRegressor_OSI.txt 2>&1 &

nohup python -u main/ViewFusionRegressor_main.py \
    --device cuda:4 --grade_type SR --num_folds 10 --start_fold 0 --num_epochs 50 --batch_size 64 \
    --lr 1e-4 --seed 42 > log/ViewFusionRegressor/ViewFusionRegressor_SR.txt 2>&1 &








