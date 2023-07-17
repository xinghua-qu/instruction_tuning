from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_asr_model(cfg):
    processor = WhisperProcessor.from_pretrained(cfg.model.asr_model_version)
    asr_model = WhisperForConditionalGeneration.from_pretrained(cfg.model.asr_model_version)
    asr_model.config.forced_decoder_ids = None
    return processor, asr_model
    
def get_llm(cfg):
    if cfg.model.llm == 'bloom':
        md_name = f'bigscience/bloomz-{cfg.model.llm_version}-mt'
        llm_tokenizer = AutoTokenizer.from_pretrained(md_name)
        llm = AutoModelForCausalLM.from_pretrained(
            md_name,
            device_map = "auto",
            torch_dtype = "auto"
        )
    if cfg.model.llm == 'falcon':
        md_name = f'tiiuae/falcon-{cfg.model.llm_version}-instruct'
        llm_tokenizer = AutoTokenizer.from_pretrained(md_name)
        llm = AutoModelForCausalLM.from_pretrained(
            md_name,
            device_map = "auto",
            torch_dtype = "auto",
            trust_remote_code=True
        )
    return llm_tokenizer, llm



