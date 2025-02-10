from transformers import  TrainingArguments, Trainer
import yaml
import argparse
from easydict import EasyDict as edict
import os
from utils import *

# set to suppress the following warning:
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# probably fine since we tokenize all data first before passing to trainer
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# hacky way to prevent loss/metrics being printed to stdout
# while still enabling logging to wandb
# https://github.com/huggingface/transformers/issues/18093
from transformers.trainer_callback import ProgressCallback
def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)
ProgressCallback.on_log = on_log


def main(configs):

    # set wandb project in which to store logs
    if 'wandb_project' in configs:
        os.environ['WANDB_PROJECT'] = configs.wandb_project


    ### Prepare Model and Tokenizer ###
    print('Preparing Model and Tokenizer')

    model = get_model(configs)
    tokenizer = get_tokenizer(configs.model_id)



    ### Prepare data ###
    print('Preparing and tokenizing data')
    t_dataset, e_dataset = get_datasets(configs, tokenizer)

    collate_fn  =  get_collate_func(tokenizer)


    ### Get custom loss objective and metrics ###
    prm_compute_loss_func = get_compute_loss_func()
    prm_compute_metrics = get_compute_metrics()
    
    ### training loop ###

    training_args = TrainingArguments(**configs.training_args)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=t_dataset,
        eval_dataset=e_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
        compute_loss_func=prm_compute_loss_func,
        compute_metrics=prm_compute_metrics
    )

    # train
    checkpoint = None
    if 'resume_from_checkpoint' in configs:
        checkpoint = configs.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script for traing Llama PRM')
    parser.add_argument('-c','--config', type=str, help='Path to config json', default='./train_configs/llama_prm800k.yml')
    args = parser.parse_args()

    with open(args.config) as stream:
        try:
            configs = edict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    
    main(configs)

