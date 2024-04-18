import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel



import subprocess

# 更新包管理器的本地数据库
subprocess.run(["apt-get", "update"])

# 安装git
subprocess.run(["apt-get", "install", "git", "-y"])

# 安装git-lfs
subprocess.run(["apt-get", "install", "git-lfs", "-y"])




# download internlm2 to the base_path directory using git tool
base_path = './assistant'
os.system(f'git clone https://code.openxlab.org.cn/nosugarcola/assistant.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-7B",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
