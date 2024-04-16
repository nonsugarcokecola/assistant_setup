import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import subprocess

def run_command(command):
    """
    执行命令并返回结果
    """
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
        return False
    return True

def update_apt_package_list():
    """
    更新APT软件包列表
    """
    print("Updating package list...")
    return run_command("sudo apt-get update")

def add_git_lfs_repository():
    """
    添加git-lfs的软件仓库
    """
    print("Adding git-lfs repository...")
    return run_command("curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash")

def install_packages(packages):
    """
    安装指定的软件包列表
    """
    print(f"Installing packages: {' '.join(packages)}")
    install_command = "sudo apt-get install -y " + " ".join(packages)
    return run_command(install_command)

def main2():
    # 更新软件包列表
    if not update_apt_package_list():
        print("Failed to update package list. Exiting.")
        return

    # 添加git-lfs仓库
    if not add_git_lfs_repository():
        print("Failed to add git-lfs repository. Exiting.")
        return

    # 安装git和git-lfs
    packages = ["git", "git-lfs"]
    if install_packages(packages):
        print("Packages installed successfully.")
    else:
        print("Failed to install packages. Exiting.")


main2()



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
