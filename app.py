import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel



import subprocess


# 使用shell=True来执行包含管道的命令
# try:
#
#
#     subprocess.run("pkill -9 apt", shell=True)
#
#     # 打印出找到的apt进程
#     # print(apt_processes)
#     # if(apt_processes.split()[0])
#     #     numb = apt_processes.split()[0]
#     # # 假设你已经找到了需要终止的apt-get进程的PID，并且存储在变量apt_pid中
#     # apt_pid = numb  # 请替换为实际的进程ID
#
#     # # 执行kill -9 {PID}来终止进程
#     # subprocess.run(f"kill -9 {apt_pid}", shell=True)
#
#     # 执行后续的apt-get命令
#     subprocess.run("apt-get update", shell=True)
#     subprocess.run("apt-get install git -y", shell=True)
#     subprocess.run("apt-get install git-lfs -y", shell=True)
#
#
# except subprocess.CalledProcessError as e:
#     print(f"An error occurred: {e}")
#
#

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
