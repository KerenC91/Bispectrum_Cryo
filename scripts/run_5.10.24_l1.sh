##################### test l1 loss ##################### (loss_method has no meaning)

####### test multiple data types  #######
# gaussian pulse
python train_main.py --N 20 --batch_size 100 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler StepLR --optimizer AdamW --lr 3e-4 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2 --data_type gaussian_pulse;

python train_main.py --N 20 --batch_size 100 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler StepLR --optimizer AdamW --lr 3e-4 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2;
####### test multiple heads       #######
python train_main.py --N 20 --batch_size 100 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler None --optimizer AdamW --lr 3e-4 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2 --data_type gaussian_pulse;

python train_main.py --N 20 --batch_size 100 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler None --optimizer AdamW --lr 3e-4 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2;
####### test multiple schedulers  #######
python train_main.py --N 20 --batch_size 100 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler None --optimizer AdamW --lr 3e-4 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2 --data_type gaussian_pulse;

python train_main.py --N 20 --batch_size 100 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler None --optimizer AdamW --lr 3e-4 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2;
####### test multiple lr          #######
python train_main.py --N 20 --batch_size 100 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler None --optimizer AdamW --lr 3e-5 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2 --data_type gaussian_pulse;

python train_main.py --N 20 --batch_size 100 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler None --optimizer AdamW --lr 3e-5 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2;
####### test multiple batch_sizes #######
python train_main.py --N 20 --batch_size 20 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler StepLR --optimizer AdamW --lr 3e-4 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2 --data_type gaussian_pulse;

python train_main.py --N 20 --batch_size 20 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler StepLR --optimizer AdamW --lr 3e-4 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2;

python train_main.py --N 20 --batch_size 5 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler StepLR --optimizer AdamW --lr 3e-4 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2 --data_type gaussian_pulse;

python train_main.py --N 20 --batch_size 5 --epochs 30000 --train_data_size 5000 --val_data_size 100 --mode rand none --n_heads 1 --loss_mode all --model 3 --scheduler StepLR --optimizer AdamW --lr 3e-4 --read_baseline 0 --comp_test_name test_l1 --comp_test_name_m baseline_K_2_N_20 --wandb --wandb_proj_name BS_inv_test_K_2 --K 2;
