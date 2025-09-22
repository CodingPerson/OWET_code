#for combineType in 'sup' 'None'
#do
#  python train.py --dataset 'BBN' --split_val 1 --train_type 'train' --cluster_rep 'vec'  --combine_type $combineType \
#  --pretrain_path 'log/BBN-22*_sup+mlm_val_HAC+HDBSCAN_aver_cp/checkpoint_BBN_lr_2e-05_latest.pth'
#done
#
#for combineType in 'sup' 'None'
#do
#  python train.py --dataset 'OntoNotes' --split_val 1 --train_type 'train' --cluster_rep 'vec'  --combine_type $combineType \
#  --pretrain_path 'log/OntoNotes-21*_sup+mlm_val_HAC+HDBSCAN_aver_cp/checkpoint_OntoNotes_lr_2e-05_latest.pth'
#done

#for combineType in 'sup' 'None'
#do
#  python train.py --dataset 'OntoNotes' --split_val 1 --train_type 'train' --cluster_rep 'box' --cluster2box 'CR_g' \
#  --enable_weight 0 --combine_type $combineType \
#  --pretrain_path 'log/OntoNotes-21*_sup+mlm_val_HAC+HDBSCAN_aver_cp/checkpoint_OntoNotes_lr_2e-05_latest.pth'
#done
#
#for cluster2box in 'R2B2' 'CR_g' 'CR_a' 'R2B'
#do
#  python train.py --dataset 'OntoNotes' --split_val 1 --train_type 'train' --cluster_rep 'box' --cluster2box $cluster2box \
#        --enable_weight 0 --combine_type 'sup' \
#        --pretrain_path 'log/OntoNotes-21*_sup+mlm_val_HAC+HDBSCAN_aver_cp/checkpoint_OntoNotes_lr_2e-05_latest.pth' \
#        --single_box_model 0 --margin=24 --lr_box 1e-3 --stop_epoch 10
#done


python train.py --dataset 'OntoNotes' --split_val 1 --train_type 'train' --cluster_rep 'box' --cluster2box 'R2B2' \
      --enable_weight 0 --combine_type 'sup' --device 0 --num_epoch 100 \
      --pretrain_path 'log/OntoNotes-_sup+mlm_val_HAC+HDBSCAN+1e-4Type_taxo_aver_cp_128/checkpoint_OntoNotes_lr_2e-05_latest.pth' \
      --single_box_model 0 --margin 20 --lr_box 1e-4 --alpha 1 --mini_ins_size 0.1 --mini_type_size 0.5 --dist_type 'BoxE' --box_self_adv 1 \
      --box_loss_type 'neg_sample' --adv_temp 16.0 --enable_box_inter 0 --enable_type_inter 0 --enable_cross_inter 0 --box_inter_type 'taxo' --inter_n_neg 0 --type_n_neg 25 --cross_n_neg 25 --proj_type2ins 0 \
      --taxo_extra 0.1 --taxo_alpha 1.0 --box_inter_start -1 --cross_inter_start -1 --taxo_eps -0.01 --taxo_margin 0.1 --type_inter_type 'taxo' --cross_inter_type 'dist' --gen_box_method 'mlp' --type_box_dim 128 \
      --proj_ins2type 1 --match_weight 0 --cross_IoU_margin 0.5 --onlyBottom 1 --use_type_emb 0 --init_type 1 --type_emb_method 'B4T' --cross_dynamic_weight 1 --type_IoU_margin 0.5 --enable_type_desc 1 \
      --cross_dist_margin 1

#for cluster2box in 'CR_g' 'CR_a' 'R2B' 'R2B2'
#do
#  for enableWeight in 0 1
#  do
#    for combineType in 'None' 'sup'
#    do
#      python train.py --dataset 'OntoNotes' --split_val 1 --train_type 'train' --cluster_rep 'box' --cluster2box $cluster2box \
#      --enable_weight $enableWeight --combine_type $combineType \
#      --pretrain_path 'log/OntoNotes-21*_sup+mlm_val_HAC+HDBSCAN_aver_cp/checkpoint_OntoNotes_lr_2e-05_latest.pth' \
#      --single_box_model 0  --lr_box 1e-3
#    done
#  done
#done

#python train.py --dataset 'BBN' --split_val 1 --train_type 'train' \
#                --pretrain_path 'log/BBN-22*_sup+mlm_val_HAC+HDBSCAN_aver_cp/checkpoint_BBN_lr_2e-05_latest.pth'
#
#python train.py --dataset 'OntoNotes' --split_val 1 --train_type 'train' \
#                --pretrain_path 'log/OntoNotes-21*_sup+mlm_val_HAC+HDBSCAN_aver_cp/checkpoint_OntoNotes_lr_2e-05_latest.pth'
#
#python train.py --dataset 'BBN' --split_val 1 --train_type 'train' --combine_type 'None'\
#                --pretrain_path 'log/BBN-22*_sup+mlm_val_HAC+HDBSCAN_aver_cp/checkpoint_BBN_lr_2e-05_latest.pth'
#
#python train.py --dataset 'OntoNotes' --split_val 1 --train_type 'train' --combine_type 'None'\
#                --pretrain_path 'log/OntoNotes-21*_sup+mlm_val_HAC+HDBSCAN_aver_cp/checkpoint_OntoNotes_lr_2e-05_latest.pth'

# sup
#python pretrain.py --dataset 'BBN' --contra_type 'test' --combine_type 'sup' --split_val 1 --train_type 'pretrain'
#
#python pretrain.py --dataset 'OntoNotes' --contra_type 'test' --combine_type 'sup' --split_val 1 --train_type 'pretrain'
#
## ce+mlm
#python pretrain.py --dataset 'BBN' --contra_type 'sup' --combine_type 'ce' --split_val 1 --train_type 'pretrain'
#
#python pretrain.py --dataset 'OntoNotes' --contra_type 'sup' --combine_type 'ce' --split_val 1 --train_type 'pretrain'
