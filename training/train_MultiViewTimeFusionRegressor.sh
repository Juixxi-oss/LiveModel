cd /data_ssd/juixxi/LiveModel
export PYTHONPATH=/data_ssd/juixxi/LiveModel
source activate env_1
clear



######################################################################################
nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type evaluator --grade_type C --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Evaluator/MultiViewTimeFusionRegressor_C.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type evaluator --grade_type N --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Evaluator/MultiViewTimeFusionRegressor_N.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type evaluator --grade_type P --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Evaluator/MultiViewTimeFusionRegressor_P.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type evaluator --grade_type BCVA --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Evaluator/MultiViewTimeFusionRegressor_BCVA.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type evaluator --grade_type MTF --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Evaluator/MultiViewTimeFusionRegressor_MTF.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type evaluator --grade_type OSI --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Evaluator/MultiViewTimeFusionRegressor_OSI.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type evaluator --grade_type SR --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Evaluator/MultiViewTimeFusionRegressor_SR.txt 2>&1 &



######################################################################################
nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type predictor --grade_type C --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Predictor/MultiViewTimeFusionRegressor_C.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type predictor --grade_type N --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Predictor/MultiViewTimeFusionRegressor_N.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type predictor --grade_type P --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Predictor/MultiViewTimeFusionRegressor_P.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type predictor --grade_type BCVA --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Predictor/MultiViewTimeFusionRegressor_BCVA.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type predictor --grade_type MTF --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Predictor/MultiViewTimeFusionRegressor_MTF.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type predictor --grade_type OSI --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Predictor/MultiViewTimeFusionRegressor_OSI.txt 2>&1 &

nohup python -u main/MultiViewTimeFusionRegressor_main.py \
    --device cuda:0 --task_type predictor --grade_type SR --num_folds 10 --start_fold 0 \
    --num_epochs 50 --batch_size 8 --lr 1e-4 --seed 42 \
    > log/MultiViewTimeFusionRegressor_Predictor/MultiViewTimeFusionRegressor_SR.txt 2>&1 &
    
    