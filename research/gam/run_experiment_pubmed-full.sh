#!/bin/bash

function main() {
  for seed in 1234 2345 9999 7839 5129; do
    for noise_level in 0 0.3 0.5; do
      local data_dir="data_${noise_level}"
      local log_dir="pubmed_full_log_${noise_level}/${seed}/"
      mkdir -p ${log_dir}
      echo "===== noise level = ${noise_level}, seed = ${seed} ====="

      date; echo "pubmed gcn_128..."
      if [ ! -f ${log_dir}/pubmed_gcn_128.log ]; then
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
            --max_num_iter_cotrain=1000 \
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
      else
        echo "    have logs before, skip...."
      fi
    done
  done
}

main "$@"
