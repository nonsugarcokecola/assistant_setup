import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import subprocess

# 使用 curl 下载安装脚本
def download_script():
    try:
        print("Installing git-lfs...")
        subprocess.run(["apt", "install", "git"], check=True)
        print("Git installation successful.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing git: {e}")


# 安装 Git LFS
def install_git_lfs():
    try:
        print("Installing git-lfs...")
        subprocess.run(["apt", "install", "git-lfs"], check=True)
        print("Git LFS installation successful.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing git-lfs: {e}")
def install_git_lfs2():
    try:
        print("Installing git-lfs...")
        subprocess.run(["git", "lfs", "install"], check=True)
        print("Git LFS installation successful2.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing git-lfs2: {e}")
# 主函数

download_script()  # 下载安装脚本
install_git_lfs()  # 安装 Git LFS
install_git_lfs2()




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
