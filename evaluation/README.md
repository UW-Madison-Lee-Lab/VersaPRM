## Get step-wise rewards from PRMs
To get the rewards on the eval data from a PRM, run
```bash
python get_rewards_reasoning_step.py \
--eval_data_dir=<dir_of_the_eval_data> \
--output_dir=<dir_to_save_the_prm_rewards> \
--eval_model_config=prm_models/model_config.json \
--prm_name=<prm_model_name> \
```
Info of supported PRMs can be found in [prm_models/model_config.json](./prm_models/model_config.json). 

## Evaluate the inference performance using PRM rewards
After getting the rewards from a PRM, to get the accuracy values and figures on the eval data, run
```bash
python calculate_metric_by_category.py \
--rewards_dir=<dir_of_the_saved_prm_rewards> \
--save_dir=<dir_to_save_the_accuracy_values_and_figures> \
--prm_name=<prm_model_name> \
--N_max=<max_num_of_generated_cots_per_question_used_for_inference> \
```