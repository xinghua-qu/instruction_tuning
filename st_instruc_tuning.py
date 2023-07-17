import torch
from datasets import load_dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, TrainingArguments, Trainer, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
from model import get_asr_model, get_llm
from utils import get_config, get_completion, set_openaikey
from evaluate import load
from jiwer import cer, wer
import opencc
from peft import LoraConfig, get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
from tqdm import tqdm
import wandb

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ast_data = load_dataset(config.data.name, config.data.version, data_dir = config.data.dir)
    ast_data = ast_data.remove_columns(['client_id', 'file', 'audio', 'id'])
    
    wandb.init(
      project="mt-peft-lora", 
      group="xinghua_ast",
      name=f"test2", 
      config={
        "learning_rate": config.trainer.lr,
        "epochs": config.trainer,
        "batch_size": config.trainer.batch_size,
      }
    )
    
    # get the tokenizer and process dataset
    model_name = f'bigscience/bloomz-{config.model.llm_version}'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        src_texts = examples["sentence"]
        tgt_texts = examples["translation"]
        inputs = tokenizer(src_texts, padding="max_length", truncation=True, max_length=config.data.max_length, return_tensors="pt")
        targets = tokenizer(tgt_texts, padding="max_length", truncation=True, max_length=config.data.max_length, return_tensors="pt")
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze(),
        }

    ast_data = ast_data.map(preprocess_function, batched = True)
    
    test_dataloader = DataLoader(
        ast_data["test"], 
        shuffle=True, 
        batch_size=128,
        collate_fn=default_data_collator
    )
    train_dataloader = DataLoader(
        ast_data["train"], 
        shuffle=True, 
        batch_size=128,
        collate_fn=default_data_collator
    )
    
    ## Get the model
    llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = "auto",
            torch_dtype = "auto"
        )
    model = llm.to(device)
    # Lora for model training
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text="Translate the following English text to Chinese:",
        tokenizer_name_or_path=model_name,
    )
    target_modules = ["query_key_value"]
    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())
    model = model.to(device)
    
    # Set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.trainer.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * config.trainer.num_epoch),
    )
    
    step = 0
    for epoch in range(config.trainer.num_epoch):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if step%10==0:
                wandb.log({"step": step, "train loss": loss})
                print(f'epoch: {epoch}, step: {step}, loss: {loss.item()}')
            step+=1
        print(f'Started the evaluation process, epoch {epoch}')
        model.eval()
        eval_loss = 0
        eval_preds = []
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )            
            wandb.log({"test loss": loss})

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
    wandb.finish()
    

#     trainer = Trainer(
#         model=model, 
#         train_dataset=ast_data['train'],
#         args=TrainingArguments(
#             per_device_train_batch_size=4, 
#             gradient_accumulation_steps=4,
#             warmup_steps=100, 
#             max_steps=200, 
#             learning_rate=2e-4, 
#             fp16=True,
#             logging_steps=1, 
#             report_to=None,
#             output_dir='outputs'
#         ),
#         data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
#         )
# #     model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
#     trainer.train()
    

if __name__ == '__main__':
    config = get_config()
    main(config)