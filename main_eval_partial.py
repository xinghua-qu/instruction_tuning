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
from jiwer import cer, wer
import opencc

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ast_data = load_dataset(config.data.name, config.data.version, data_dir = config.data.dir, split='test')
    print(ast_data)
    
    # get the model
    processor, asr_model = get_asr_model(config)
    asr_model = asr_model.to(device)
    llm_tokenizer, llm = get_llm(config)
    llm = llm.to(device)
    converter = opencc.OpenCC('t2s.json')
    
    def map_to_pred(data_item):
        audio = data_item["audio"]
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        if len(data_item['translation'])==0:
            data_item["reference"] = ''
            data_item["prediction"] = ''
            print('#### Empty string')
            return None
        data_item["reference"] = processor.tokenizer._normalize(data_item['translation'])

        with torch.no_grad():
            predicted_ids = asr_model.generate(input_features.to(device))[0]
            transcription = processor.decode(predicted_ids)
            transcription = processor.tokenizer._normalize(transcription)
            data_item["sentence"] = processor.tokenizer._normalize(data_item["sentence"])
            if len(transcription)>=config.model.max_length:
                transcription = transcription[0:config.model.max_length]
            
            if config.model.api:
                set_openaikey()
                prompt = "Translate the following English text into Chinese. Directly output the translated Chinese. ###{}###"
                translations = get_completion(prompt.format(transcription))
            else:
                prompt = config.model.prompt
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
                    pad_token_id=llm_tokenizer.eos_token_id,
                )
                translations = llm_tokenizer.batch_decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
        
            translations = ''.join(translations).replace(prompt.format(transcription),"")
        print(translations)
        data_item["prediction"] = processor.tokenizer._normalize(translations)
        if config.model.llm=='bloom':
            data_item["prediction"] = converter.convert(data_item["prediction"])
        print(data_item["sentence"], '<|>',transcription)
        print(data_item["reference"], '<|>',data_item["prediction"])
        asr_wer_val = wer(data_item["sentence"], transcription)        
        trans_cer_val = cer(data_item["reference"], data_item["prediction"])
        print(asr_wer_val, trans_cer_val)
        return data_item, asr_wer_val, trans_cer_val, data_item["reference"], data_item["prediction"]
    
    trans_mtk = []
    asr_mtk = []
    references = []
    predictions = []
    for i in range(config.data.eval_num):
        print(f'#### item {i}')
        data_item =  ast_data[i]    
        _, asr_wer_val, trans_cer_val, itm_ref, itm_pre = map_to_pred(data_item)    
        trans_mtk.append(trans_cer_val)
        asr_mtk.append(asr_wer_val)
        references.append(itm_ref)
        predictions.append(itm_pre)
        print(asr_wer_val, trans_cer_val)

    bleu = load("sacrebleu")
    bleu_scores = bleu.compute(predictions=predictions, references=references)
    bleu_val = bleu_scores['score']
    print(config)
    print(f'Bleu score: {bleu_val}')
    print(f'translation cer mean {sum(trans_mtk)/len(trans_mtk)}')
    print(f'asr wer mean {sum(asr_mtk)/len(asr_mtk)}')                                     
    return None

if __name__ == '__main__':
    config = get_config()
    main(config)