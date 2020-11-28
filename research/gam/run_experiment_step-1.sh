#!/bin/bash

function main() {
  for step in 1; do
    echo "=========== cotrain step num: ${step}"
    for seed in 1234 2345 9999 7839 5129 3584 4976 2360 1112 7713; do
      # for reg_lu in 0.1 1 10 100 1000 10000; do
      local reg_lu=100
      local reg_uu=$(python3 -c "print(0.5 * ${reg_lu})")

      for noise_level in 0 0.3 0.5; do
        local data_dir="data_${noise_level}"
        local log_dir="log_${noise_level}/${step}/${seed}/"
        mkdir -p ${log_dir}
        echo "===== noise level = ${noise_level}, seed = ${seed} ====="

        date; echo "cora gcn_16..."
        if [ ! -f ${log_dir}/cora_gcn_16_reg-${reg_lu}.log ]; then
          python3 -m gam.experiments.run_train_gam_graph \
              --data_path=${data_dir}/ \
              --dataset_name=cora \
              --model_cls=gcn \
              --model_agr=mlp \
              --hidden_cls=16 \
              --hidden_agr=128 \
              --num_pairs_reg=512 \
              --first_iter_original=True \
              --add_negative_edges_agr=True \
              --aggregation_agr_inputs=dist \
              --num_samples_to_label=200  \
              --max_num_iter_cotrain=${step} \
              --first_iter_original=True \
              --learning_rate_cls=1e-3 \
              --num_iter_after_best_val_agr=2000 \
              --experiment_suffix='-GAM' \
              --num_iter_after_best_val_cls=2000 \
              --row_normalize=False \
              --reg_weight_lu=${reg_lu} \
              --reg_weight_uu=${reg_uu} \
              --always_agree=False \
              --keep_label_proportions=True \
              --seed=${seed} \
              --use_graph=True \
              --penalize_neg_agr=False \
              --reg_weight_vat=0 \
              2> ${log_dir}/cora_gcn_16_reg-${reg_lu}.log
        else
          echo "    have logs before, skip...."
        fi

        date; echo "citeseer gcn_16..."
        if [ ! -f ${log_dir}/citeseer_gcn_16_reg-${reg_lu}.log ]; then
          python3 -m gam.experiments.run_train_gam_graph \
              --data_path=${data_dir}/ \
              --dataset_name=citeseer  \
              --model_cls=gcn \
              --model_agr=mlp \
              --hidden_cls=16 \
              --hidden_agr=128 \
              --batch_size_cls=128 \
              --batch_size_agr=128  \
              --num_pairs_reg=512 \
              --first_iter_original=True \
              --add_negative_edges_agr=True  \
              --aggregation_agr_inputs=dist  \
              --num_samples_to_label=200  \
              --max_num_iter_cotrain=${step} \
              --experiment_suffix='-GAM'   \
              --first_iter_original=True \
              --learning_rate_cls=1e-3 \
              --num_iter_after_best_val_agr=2000 \
              --num_iter_after_best_val_cls=2000 \
              --row_normalize=False \
              --reg_weight_lu=${reg_lu} \
              --reg_weight_uu=${reg_uu} \
              --always_agree=False \
              --seed=${seed} \
              --keep_label_proportions=True \
              --use_graph=True \
              --penalize_neg_agr=False \
              --reg_weight_vat=0 \
              2> ${log_dir}/citeseer_gcn_16_reg-${reg_lu}.log
        else
          echo "    have logs before, skip...."
        fi

        date; echo "pubmed gcn_16..."
        if [ ! -f ${log_dir}/pubmed_gcn_16_reg-${reg_lu}.log ]; then
          python3 -m gam.experiments.run_train_gam_graph \
              --data_path=${data_dir}/ \
              --dataset_name=pubmed  \
              --model_cls=gcn \
              --model_agr=mlp \
              --hidden_cls=16 \
              --hidden_agr=128 \
              --batch_size_cls=128 \
              --batch_size_agr=128  \
              --num_pairs_reg=512 \
              --first_iter_original=True \
              --add_negative_edges_agr=True  \
              --aggregation_agr_inputs=dist  \
              --num_samples_to_label=200  \
              --max_num_iter_cotrain=${step} \
              --experiment_suffix='-GAM'   \
              --first_iter_original=True \
              --learning_rate_cls=1e-3 \
              --num_iter_after_best_val_agr=2000 \
              --num_iter_after_best_val_cls=2000 \
              --row_normalize=False \
              --reg_weight_lu=${reg_lu} \
              --reg_weight_uu=${reg_uu} \
              --always_agree=False \
              --seed=${seed} \
              --keep_label_proportions=False \
              2> ${log_dir}/pubmed_gcn_16_reg-${reg_lu}.log
        else
          echo "    have logs before, skip...."
        fi
      done
    done
  done
}

main "$@"
