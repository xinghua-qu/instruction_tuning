import yaml
import argparse
import openai
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", required=True, help="path to configuration file.")
    args = parser.parse_args()
    return args

def get_config():
    args = parse_args()
    config = OmegaConf.load(args.config)
#     config = OmegaConf.merge(args, read_config)
    return config


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def set_openaikey():
    openai.api_key  = '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'
    return None