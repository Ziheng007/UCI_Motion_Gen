import os
import openai
import yaml
import re
import json
import re
with open("/extra/xielab0/haoyum3/Ask-Anything/videochat_finetue/llm_utils/config.yaml") as f:
 config_yaml = yaml.load(f, Loader=yaml.FullLoader)

client = openai.OpenAI(api_key=config_yaml['token'])

MODEL = "gpt-4o"

def llm(prompt, stop=["\n"]):

    dialogs = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=dialogs
    )
    return completion.choices[0].message.content


