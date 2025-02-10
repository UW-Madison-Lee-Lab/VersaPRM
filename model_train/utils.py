from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from prm_datasets import TokenizedPRMDataset
import evaluate
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import softmax


def get_model(configs):
    

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)


    if 'lora_config' in configs:
        print('Using LoRA')
        lora_config = LoraConfig(**configs.lora_config)
        model = get_peft_model(model, lora_config)
        
    return model

def get_tokenizer(model_id):
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token #llama doesn't define pad token, so we need to do this
    tokenizer.padding_side='right' # we need to pad from right (so that we can do eval mask id trick for eval)


    return tokenizer

def get_datasets(configs, tokenizer):
    
    t_dataset = TokenizedPRMDataset(configs.train_data_path, 
                                    tokenizer,
                                    label_last_n = configs.train_label_last_n if 'train_label_last_n' in configs else None,
                                    max_length=configs.max_length if 'max_length' in configs else None,
                                    use_augs=configs.use_augs if 'use_augs' in configs else True)
    e_dataset = TokenizedPRMDataset(configs.eval_data_path, 
                                    tokenizer,
                                    label_last_n = configs.eval_label_last_n if 'eval_label_last_n' in configs else None,
                                    max_length=configs.max_length if 'max_length' in configs else None,
                                    use_augs=configs.use_augs if 'use_augs' in configs else True) if configs.eval_data_pat is not None else None
    return t_dataset, e_dataset

def get_collate_func(tokenizer):
      
    return DataCollatorForTokenClassification(tokenizer=tokenizer, 
                                                        padding='longest', 
                                                        label_pad_token_id=-100,
                                                        return_tensors='pt')


def get_compute_loss_func():
      
    def compute_loss_func(outputs, labels, num_items_in_batch):


        # output logits are in shape (B, L, V) - batch, seq length, vocab size

        # 12 is ID of '-', 10 is ID of '+' (for both Llama and Qwen tokenizer)
        # TODO: change so its more flexible for different tokenizers
        logits = outputs.logits[:,:,[12,10]].reshape(-1,2)


        # for eval, num_items_in_batch is None
        if num_items_in_batch is None:
            loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100)
            return loss

        # num_items_in_batch
        # https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/trainer.py#L5142
        loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100,
                            reduction='sum')

        return loss / num_items_in_batch
    
    return compute_loss_func


def get_compute_metrics():
    '''
    gets metrics for precision, recall, f1 score...
    '''
       
    
    accuracy = evaluate.load('accuracy')
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')
    f1 = evaluate.load('f1')


    def compute_metrics(eval_pred):
        logits, labels = eval_pred


        label_mask_PRM = (labels!=-100)

        labels_PRM = labels[label_mask_PRM]
        logits_PRM = logits[:,:,[12, 10]][label_mask_PRM]

        pred_PRM = np.argmax(logits_PRM, axis=-1)
        predf_PRM = softmax(logits_PRM)[:,1]


        results = {
            'PRM Accuracy': accuracy.compute(predictions=pred_PRM, references=labels_PRM)['accuracy'],
            'PRM Precision': precision.compute(predictions=pred_PRM, references=labels_PRM, zero_division=0.0)['precision'],
            'PRM Recall': recall.compute(predictions=pred_PRM, references=labels_PRM)['recall'],
            'PRM Specificty': recall.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['recall'],
            'PRM NPV': precision.compute(predictions=pred_PRM, references=labels_PRM, pos_label= 0, zero_division=0.0)['precision'], # negative predictive value, unPrecision
            'PRM F1': f1.compute(predictions=pred_PRM, references=labels_PRM)['f1'],
            'PRM F1 Neg': f1.compute(predictions=pred_PRM, references=labels_PRM, pos_label=0)['f1'],
            'PRM F1 AUC': roc_auc_score(labels_PRM, predf_PRM),
            }
    

        return results
    
    return compute_metrics