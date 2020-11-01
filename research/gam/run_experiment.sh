#!/bin/bash

function main() {
  for seed in 1234 2345 9999 7839 5129 3583 4975 2359 1111 7712; do
    for noise_level in 0 0.3 0.5; do 
      local data_dir="data_${noise_level}"
      local log_dir="log_${noise_level}/${seed}/"
      mkdir -p ${log_dir}
      echo "===== noise level = ${noise_level}, seed = ${seed} ====="

      date; echo "cora mlp_128..."
      python3 -m gam.experiments.run_train_gam_graph \
          --data_path=${data_dir}/ \
          --dataset_name=cora  \
          --model_cls=mlp \
          --model_agr=mlp \
          --hidden_cls=128 \
          --hidden_agr=128 \
          --batch_size_cls=128 \
          --batch_size_agr=128 \
          --num_pairs_reg=128 \
          --first_iter_original=True \
          --add_negative_edges_agr=True \
          --aggregation_agr_inputs=dist \
          --num_samples_to_label=200 \
          --max_num_iter_cotrain=30 \
          --experiment_suffix='-GAM' \
          --first_iter_original=True \
          --learning_rate_cls=1e-3 \
          --num_iter_after_best_val_agr=2000 \
          --num_iter_after_best_val_cls=2000 \
          --row_normalize=False --reg_weight_lu=0.1 \
          --reg_weight_uu=0.05 \
          --always_agree=False \
          --seed=${seed} \
          --keep_label_proportions=True \
          --use_graph=True \
          --penalize_neg_agr=False \
          2> ${log_dir}/cora_mlp_128.log

      date; echo "cora mlp_32_32_32_32..."
      python3 -m gam.experiments.run_train_gam_graph \
          --data_path=${data_dir}/ \
          --dataset_name=cora \
          --model_cls=mlp \
          --model_agr=mlp \
          --hidden_cls=32_32_32_32 \
          --hidden_agr=32_32_32_32 \
          --batch_size_cls=128 \
          --batch_size_agr=128 \
          --num_pairs_reg=128 \
          --first_iter_original=True \
          --add_negative_edges_agr=True \
          --aggregation_agr_inputs=dist \
          --num_samples_to_label=200 \
          --max_num_iter_cotrain=30 \
          --experiment_suffix='-GAM' \
          --first_iter_original=True \
          --learning_rate_cls=1e-3 \
          --num_iter_after_best_val_agr=2000 \
          --num_iter_after_best_val_cls=2000 \
          --row_normalize=False \
          --reg_weight_lu=1000 \
          --reg_weight_uu=500 \
          --always_agree=False \
          --seed=${seed} \
          --keep_label_proportions=True \
          --use_graph=True \
          --penalize_neg_agr=False \
          2> ${log_dir}/cora_mlp_32_32_32_32.log

      date; echo "cora gcn_128..."
      python3 -m gam.experiments.run_train_gam_graph \
          --data_path=${data_dir}/ \
          --dataset_name=cora \
          --model_cls=gcn \
          --model_agr=mlp \
          --hidden_cls=128 \
          --hidden_agr=128 \
          --num_pairs_reg=512 \
          --first_iter_original=True \
          --add_negative_edges_agr=True \
          --aggregation_agr_inputs=dist \
          --num_samples_to_label=200  \
          --max_num_iter_cotrain=30 \
          --first_iter_original=True \
          --learning_rate_cls=1e-3 \
          --num_iter_after_best_val_agr=2000 \
          --experiment_suffix='-GAM' \
          --num_iter_after_best_val_cls=2000 \
          --row_normalize=False \
          --reg_weight_lu=100 \
          --reg_weight_uu=50 \
          --always_agree=False \
          --keep_label_proportions=True \
          --seed=${seed} \
          --use_graph=True \
          --penalize_neg_agr=False \
          --reg_weight_vat=0 \
          2> ${log_dir}/cora_gcn_128.log

      date; echo "citeseer mlp_128..."
      python3 -m gam.experiments.run_train_gam_graph \
          --data_path=${data_dir}/ \
          --dataset_name=citeseer  \
          --model_cls=mlp \
          --model_agr=mlp \
          --hidden_cls=128 \
          --hidden_agr=128 \
          --batch_size_cls=128 \
          --batch_size_agr=128  \
          --num_pairs_reg=128 \
          --first_iter_original=True \
          --add_negative_edges_agr=True  \
          --aggregation_agr_inputs=dist  \
          --num_samples_to_label=200  \
          --max_num_iter_cotrain=30 \
          --experiment_suffix='-GAM'   \
          --first_iter_original=True \
          --learning_rate_cls=1e-3 \
          --num_iter_after_best_val_agr=2000 \
          --num_iter_after_best_val_cls=2000 \
          --row_normalize=False \
          --reg_weight_lu=10.0 \
          --reg_weight_uu=5 \
          --always_agree=False \
          --seed=${seed} \
          --keep_label_proportions=True \
          --use_graph=True \
          --penalize_neg_agr=False \
          2> ${log_dir}/citeseer_mlp_128.log

      date; echo "citeseer mlp_32_32_32_32..."
      python3 -m gam.experiments.run_train_gam_graph \
          --data_path=${data_dir}/ \
          --dataset_name=citeseer  \
          --model_cls=mlp \
          --model_agr=mlp \
          --hidden_cls=32_32_32_32 \
          --hidden_agr=32_32_32_32 \
          --batch_size_cls=128 \
          --batch_size_agr=128  \
          --num_pairs_reg=512 \
          --first_iter_original=True \
          --add_negative_edges_agr=True  \
          --aggregation_agr_inputs=dist  \
          --num_samples_to_label=200  \
          --max_num_iter_cotrain=30 \
          --experiment_suffix='-GAM'  \
          --first_iter_original=True \
          --learning_rate_cls=1e-3 \
          --num_iter_after_best_val_agr=2000 \
          --num_iter_after_best_val_cls=2000 \
          --row_normalize=False \
          --reg_weight_lu=10 \
          --reg_weight_uu=5 \
          --always_agree=False \
          --seed=${seed} \
          --keep_label_proportions=True \
          --use_graph=True \
          --penalize_neg_agr=False \
          2> ${log_dir}/citeseer_mlp_32_32_32_32.log

      date; echo "citeseer gcn_128..."
      python3 -m gam.experiments.run_train_gam_graph \
          --data_path=${data_dir}/ \
          --dataset_name=citeseer  \
          --model_cls=gcn \
          --model_agr=mlp \
          --hidden_cls=128 \
          --hidden_agr=128 \
          --batch_size_cls=128 \
          --batch_size_agr=128  \
          --num_pairs_reg=512 \
          --first_iter_original=True \
          --add_negative_edges_agr=True  \
          --aggregation_agr_inputs=dist  \
          --num_samples_to_label=200  \
          --max_num_iter_cotrain=30 \
          --experiment_suffix='-GAM'   \
          --first_iter_original=True \
          --learning_rate_cls=1e-3 \
          --num_iter_after_best_val_agr=2000 \
          --num_iter_after_best_val_cls=2000 \
          --row_normalize=False \
          --reg_weight_lu=100 \
          --reg_weight_uu=50 \
          --always_agree=False \
          --seed=${seed} \
          --keep_label_proportions=True \
          --use_graph=True \
          --penalize_neg_agr=False \
          --reg_weight_vat=0 \
          2> ${log_dir}/citeseer_gcn_128.log

      date; echo "pubmed mlp_128..."
      python3 -m gam.experiments.run_train_gam_graph \
          --data_path=${data_dir}/ \
          --dataset_name=pubmed \
          --model_cls=mlp \
          --model_agr=mlp \
          --hidden_cls=128 \
          --hidden_agr=128 \
          --batch_size_cls=128 \
          --batch_size_agr=128 \
          --num_pairs_reg=128 \
          --first_iter_original=True \
          --add_negative_edges_agr=True \
          --aggregation_agr_inputs=dist \
          --num_samples_to_label=200 \
          --max_num_iter_cotrain=30 \
          --experiment_suffix='-GAM' \
          --first_iter_original=True \
          --learning_rate_cls=1e-3 \
          --num_iter_after_best_val_agr=2000 \
          --num_iter_after_best_val_cls=2000 \
          --row_normalize=False \
          --reg_weight_lu=10 \
          --reg_weight_uu=5 \
          --always_agree=False \
          --seed=${seed} \
          --keep_label_proportions=False \
          --use_graph=True \
          --penalize_neg_agr=False \
          2> ${log_dir}/pubmed_mlp_128.log

      date; echo "pubmed mlp_32_32_32_32..."
      python3 -m gam.experiments.run_train_gam_graph \
          --data_path=${data_dir}/ \
          --dataset_name=pubmed  \
          --model_cls=mlp \
          --model_agr=mlp \
          --hidden_cls=32_32_32_32 \
          --hidden_agr=32_32_32_32 \
          --batch_size_cls=128 \
          --batch_size_agr=128  \
          --num_pairs_reg=512 \
          --first_iter_original=True \
          --add_negative_edges_agr=True  \
          --aggregation_agr_inputs=dist  \
          --num_samples_to_label=200  \
          --max_num_iter_cotrain=30 \
          --experiment_suffix='-GAM'  \
          --first_iter_original=True \
          --learning_rate_cls=1e-3 \
          --num_iter_after_best_val_agr=2000 \
          --num_iter_after_best_val_cls=2000 \
          --row_normalize=False \
          --reg_weight_lu=1000 \
          --reg_weight_uu=500 \
          --always_agree=False \
          --seed=${seed} \
          --keep_label_proportions=True \
          --use_graph=True \
          --penalize_neg_agr=False \
          --reg_weight_vat=0 \
          2> ${log_dir}/pubmed_mlp_32_32_32_32.log

      date; echo "pubmed gcn_128..."
      python3 -m gam.experiments.run_train_gam_graph \
          --data_path=${data_dir}/ \
          --dataset_name=pubmed  \
          --model_cls=gcn \
          --model_agr=mlp \
          --hidden_cls=128 \
          --hidden_agr=128 \
          --batch_size_cls=128 \
          --batch_size_agr=128  \
          --num_pairs_reg=512 \
          --first_iter_original=True \
          --add_negative_edges_agr=True  \
          --aggregation_agr_inputs=dist  \
          --num_samples_to_label=200  \
          --max_num_iter_cotrain=30 \
          --experiment_suffix='-GAM'   \
          --first_iter_original=True \
          --learning_rate_cls=1e-3 \
          --num_iter_after_best_val_agr=2000 \
          --num_iter_after_best_val_cls=2000 \
          --row_normalize=False \
          --reg_weight_lu=100.0 \
          --reg_weight_uu=50 \
          --always_agree=False \
          --seed=${seed} \
          --keep_label_proportions=False \
          2> ${log_dir}/pubmed_gcn_128.log
    done
  done
}

main "$@"
