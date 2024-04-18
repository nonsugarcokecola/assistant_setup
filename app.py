import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel



import subprocess

# 执行ps -e | grep apt来找到apt-get的进程
result = subprocess.run(["ps", "-e", "|", "grep", "apt"], stdout=subprocess.PIPE, text=True)
apt_processes = result.stdout.splitlines()

# 假设我们要终止所有名为apt的进程
for proc in apt_processes:
    try:
        # 提取进程ID（这里假设输出格式为：procID  pts/0    Ss   0:00 /bin/apt）
        proc_id = proc.split()[1]
        # 执行kill -9 {proc_id}来终止进程
        subprocess.run(["kill", "-9", proc_id])
        print(f"Killed process with ID: {proc_id}")
    except (IndexError, ValueError):
        # 如果无法解析进程ID，或者进程已经终止，就忽略它
        pass

# 执行apt-get update来更新软件包列表
subprocess.run(["apt-get", "update"])

# 执行apt-get install git来安装git
subprocess.run(["apt-get", "install", "git", "-y"])

# 执行apt-get install git-lfs来安装git-lfs
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
