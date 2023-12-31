from datasets import load_dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from model import get_asr_model, get_llm
from transformers import TrainingArguments, Trainer
from utils import get_config, get_completion, set_openaikey
import torch
from evaluate import load

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ast_data = load_dataset(config.data.name, config.data.version, data_dir = config.data.dir, split='test')
    print(ast_data)
    for i in range(10):
        print(ast_data[i]['translation'])
    exit(0)
    
#     test_dataloader = DataLoader(
#         ast_data["test"], 
#         shuffle=True, 
#         batch_size=8,
#         collate_fn=collate_fn
#         )
    
    # get the model
    processor, asr_model = get_asr_model(config)
    asr_model = asr_model.to(device)
    llm_tokenizer, llm = get_llm(config)
    llm = llm.to(device)
    
    def map_to_pred(batch):
        audio = batch["audio"]
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        if len(batch['translation'])==0:
            batch["reference"] = ''
            batch["prediction"] = ''
            print('#### Empty string')
            return None
        batch["reference"] = processor.tokenizer._normalize(batch['translation'])

        with torch.no_grad():
            predicted_ids = asr_model.generate(input_features.to(device))[0]
            transcription = processor.decode(predicted_ids)
            transcription = processor.tokenizer._normalize(transcription)
            if len(transcription)>=config.model.max_length:
                transcription = transcription[0:config.model.max_length]
            print('### Input: {}'.format(transcription))
            if config.model.api:
                set_openaikey()
                prompt = "Translate the following English text into Chinese. Directly output the translated Chinese. ###{}###"
                print(prompt.format(transcription))
                translations = get_completion(prompt.format(transcription))
                print('### Output: {}'.format(translations), batch["reference"])
                exit(0)
            else:
#                 prompt = "To say '{}' in Chinese, you would say"
                prompt = "Translate the following English text into Chinese. {}"
                inputs = llm_tokenizer(prompt.format(transcription), return_tensors="pt")
                outputs = llm.generate(
                    inputs["input_ids"], 
                    attention_mask= inputs["attention_mask"], 
                    max_length=config.model.max_length,
                    num_beams=1, 
                    no_repeat_ngram_size=2, 
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                translations = llm_tokenizer.batch_decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
                print('### Output: {}'.format(''.join(translations)))
            batch["prediction"] = processor.tokenizer._normalize(''.join(translations))
        return batch
    
    result = ast_data.map(map_to_pred)

    wer = load("wer")
    print(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))
    return None

if __name__ == '__main__':
    config = get_config()
    main(config)
                                              