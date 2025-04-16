"""
Altered from webui.py
"""

import os
import sys

# if len(sys.argv) == 1:
#     sys.argv.append("v2")
# version = "v1" if sys.argv[1] == "v1" else "v2"
# os.environ["version"] = version

from pathlib import Path
now_dir = Path(__file__).absolute().parent.as_posix()
gptsovits_dir = Path(now_dir).joinpath("GPT_SoVITS").as_posix()
sys.path.insert(0, f"{gptsovits_dir}")
sys.path.insert(1, f"{now_dir}")
os.chdir(now_dir)

import warnings
warnings.filterwarnings("ignore")
import json
import platform
import re
import shutil
import signal
import psutil
import torch
import yaml
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if os.path.exists(tmp):
    for name in os.listdir(tmp):
        if name == "jieba.cache":
            continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import traceback
import subprocess
from subprocess import Popen
from config import (
    exp_root,
    infer_device,
    is_half,
    is_share,
    python_exec,
    webui_port_infer_tts,
)
#from GPT_SoVITS.tools import my_utils
from GPT_SoVITS.tools.i18n.i18n import I18nAuto, scan_language_list
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
os.environ["language"] = language
i18n = I18nAuto(language=language)
from multiprocessing import cpu_count


n_cpu = cpu_count()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 判断是否有能用来训练和加速推理的N卡
ok_gpu_keywords = {
    "10",
    "16",
    "20",
    "30",
    "40",
    "A2",
    "A3",
    "A4",
    "P4",
    "A50",
    "500",
    "A60",
    "70",
    "80",
    "90",
    "M4",
    "T4",
    "TITAN",
    "L4",
    "4060",
    "H",
    "600",
    "506",
    "507",
    "508",
    "509",
}
set_gpu_numbers = set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))
# # 判断是否支持mps加速
# if torch.backends.mps.is_available():
#     if_gpu_ok = True
#     gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
#     mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # 实测使用系统内存作为显存不会爆显存


def set_default(version):
    global \
        default_batch_size, \
        default_max_batch_size, \
        gpu_info, \
        default_sovits_epoch, \
        default_sovits_save_every_epoch, \
        max_sovits_epoch, \
        max_sovits_save_every_epoch, \
        default_batch_size_s1, \
        if_force_ckpt
    if_force_ckpt = False
    if if_gpu_ok and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
        minmem = min(mem)
        # if version == "v3" and minmem < 14:
        #     # API读取不到共享显存,直接填充确认
        #     try:
        #         torch.zeros((1024,1024,1024,14),dtype=torch.int8,device="cuda")
        #         torch.cuda.empty_cache()
        #         minmem = 14
        #     except RuntimeError as _:
        #         # 强制梯度检查只需要12G显存
        #         if minmem >= 12 :
        #             if_force_ckpt = True
        #             minmem = 14
        #         else:
        #             try:
        #                 torch.zeros((1024,1024,1024,12),dtype=torch.int8,device="cuda")
        #                 torch.cuda.empty_cache()
        #                 if_force_ckpt = True
        #                 minmem = 14
        #             except RuntimeError as _:
        #                 print("显存不足以开启V3训练")
        default_batch_size = minmem // 2 if version != "v3" else minmem // 8
        default_batch_size_s1 = minmem // 2
    else:
        gpu_info = "%s\t%s" % ("0", "CPU")
        gpu_infos.append("%s\t%s" % ("0", "CPU"))
        set_gpu_numbers.add(0)
        default_batch_size = default_batch_size_s1 = int(psutil.virtual_memory().total / 1024 / 1024 / 1024 / 4)
    if version != "v3":
        default_sovits_epoch = 8
        default_sovits_save_every_epoch = 4
        max_sovits_epoch = 25  # 40
        max_sovits_save_every_epoch = 25  # 10
    else:
        default_sovits_epoch = 2
        default_sovits_save_every_epoch = 1
        max_sovits_epoch = 3  # 40
        max_sovits_save_every_epoch = 3  # 10

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3


gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers = str(sorted(list(set_gpu_numbers))[0])


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


p_tts_inference = None


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


system = platform.system()


def kill_process(pid, process_name=""):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        # os.system(cmd)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        kill_proc_tree(pid)
    print(process_name + i18n("进程已终止"))


def process_info(process_name="", indicator=""):
    if indicator == "opened":
        return process_name + i18n("已开启")
    elif indicator == "open":
        return i18n("开启") + process_name
    elif indicator == "closed":
        return process_name + i18n("已关闭")
    elif indicator == "close":
        return i18n("关闭") + process_name
    elif indicator == "running":
        return process_name + i18n("运行中")
    elif indicator == "occupy":
        return process_name + i18n("占用中") + "," + i18n("需先终止才能开启下一次任务")
    elif indicator == "finish":
        return process_name + i18n("已完成")
    elif indicator == "failed":
        return process_name + i18n("失败")
    elif indicator == "info":
        return process_name + i18n("进程输出信息")
    else:
        return process_name


def runCMD(args: str):
    encoding = 'gbk' if system else 'utf-8'
    totalInput = f"{args}\n".encode(encoding)
    if system == 'Windows':
        shellArgs = ['cmd']
    if system == 'Linux':
        shellArgs = ['bash', '-c']
    subproc = subprocess.Popen(
        args = shellArgs,
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT
    )
    totalOutput = subproc.communicate(totalInput)[0]
    return '' if totalOutput is None else totalOutput.strip().decode(encoding)


netStat = runCMD(f'netstat -aon|findstr "{webui_port_infer_tts}"')
for line in str(netStat).splitlines():
    line = line.strip()
    runCMD(f'taskkill /T /F /PID {line.split(" ")[-1]}') if line.startswith("TCP") else None


process_name_tts = i18n("TTS推理WebUI")


def change_tts_inference(
    bert_path,
    cnhubert_base_path,
    gpu_number,
    gpt_path,
    sovits_path,
    sovits_v3_path,
    bigvgan_path,
    is_half,
    batched_infer_enabled
):
    global p_tts_inference
    if batched_infer_enabled:
        cmd = '"%s" GPT_SoVITS/inference_webui_fast.py "%s"' % (python_exec, language)
    else:
        cmd = '"%s" GPT_SoVITS/inference_webui.py "%s"' % (python_exec, language)
    # #####v3暂不支持加速推理
    # if version=="v3":
    #     cmd = '"%s" GPT_SoVITS/inference_webui.py "%s"'%(python_exec, language)
    if p_tts_inference is None:
        os.environ["gpt_path"] = gpt_path# if "/" in gpt_path else "%s/%s" % (GPT_weight_root, gpt_path)
        os.environ["sovits_path"] = sovits_path# if "/" in sovits_path else "%s/%s" % (SoVITS_weight_root, sovits_path)
        os.environ["sovits_v3_path"] = sovits_v3_path
        os.environ["cnhubert_base_path"] = cnhubert_base_path
        os.environ["bert_path"] = bert_path
        os.environ["bigvgan_path"] = bigvgan_path
        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_number(gpu_number)
        os.environ["is_half"] = str(is_half)
        os.environ["infer_ttswebui"] = str(webui_port_infer_tts)
        os.environ["is_share"] = str(is_share)
        # yield (
        #     process_info(process_name_tts, "opened"),
        #     {"__type__": "update", "visible": False},
        #     {"__type__": "update", "visible": True},
        # )
        print("TTS推理进程已开启")
        print(cmd)
        p_tts_inference = Popen(cmd, shell=True)
        p_tts_inference.wait()
    else:
        kill_process(p_tts_inference.pid, process_name_tts)
        p_tts_inference = None
        # yield (
        #     process_info(process_name_tts, "closed"),
        #     {"__type__": "update", "visible": True},
        #     {"__type__": "update", "visible": False},
        # )
        print("TTS推理进程已关闭")


def convert(
    version: str = "v3",
    sovits_path: str = ...,
    sovits_v3_path: str = ...,
    gpt_path: str = ...,
    cnhubert_base_path: str = ...,
    bert_path: str = ...,
    bigvgan_path: str = ...,
    half_precision: bool = True,
    batched_infer: bool = False,
):
    os.environ["version"] = version
    set_default(version)
    # 1C-推理
    change_tts_inference(
        bert_path = bert_path,
        cnhubert_base_path = cnhubert_base_path,
        gpu_number = gpus,
        gpt_path = gpt_path,
        sovits_path = sovits_path,
        sovits_v3_path = sovits_v3_path,
        bigvgan_path = bigvgan_path,
        is_half = half_precision,
        batched_infer_enabled = batched_infer,
    )

    # 2-GPT-SoVITS-变声