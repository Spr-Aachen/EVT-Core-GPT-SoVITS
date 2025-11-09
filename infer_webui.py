"""
Altered from webui.py
"""

import os
import sys
#os.environ["version"] = version = "v2Pro"
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
import site
import traceback
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if site_packages_roots == []:
    site_packages_roots = ["%s/runtime/Lib/site-packages" % now_dir]
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    # "%s\n%s/runtime\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    "%s\n%s/GPT_SoVITS/BigVGAN\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError:
            traceback.print_exc()
import shutil
import subprocess
from subprocess import Popen

from GPT_SoVITS.tools.i18n.i18n import I18nAuto, scan_language_list

language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
os.environ["language"] = language
i18n = I18nAuto(language=language)
from multiprocessing import cpu_count

from config import (
    GPU_INDEX,
    GPU_INFOS,
    IS_GPU,
    exp_root,
    infer_device,
    is_half,
    is_share,
    memset,
    python_exec,
    webui_port_infer_tts,
    webui_port_main,
    webui_port_subfix,
    webui_port_uvr5,
)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

n_cpu = cpu_count()

set_gpu_numbers = GPU_INDEX
gpu_infos = GPU_INFOS
mem = memset
is_gpu_ok = IS_GPU

v3v4set = {"v3", "v4"}


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
    gpu_info = "\n".join(gpu_infos)
    if is_gpu_ok:
        minmem = min(mem)
        default_batch_size = minmem // 2 if version not in v3v4set else minmem // 8
        default_batch_size_s1 = minmem // 2
    else:
        default_batch_size = default_batch_size_s1 = int(psutil.virtual_memory().total / 1024 / 1024 / 1024 / 4)
    if version not in v3v4set:
        default_sovits_epoch = 8
        default_sovits_save_every_epoch = 4
        max_sovits_epoch = 25  # 40
        max_sovits_save_every_epoch = 25  # 10
    else:
        default_sovits_epoch = 2
        default_sovits_save_every_epoch = 1
        max_sovits_epoch = 16  # 40 # 3 #训太多=作死
        max_sovits_save_every_epoch = 10  # 10 # 3

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3


gpus = "-".join(map(str, GPU_INDEX))
default_gpu_numbers = infer_device.index


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
    sovits_v4_path,
    bigvgan_path,
    vocoder_path,
    sv_path,
    g2pw_path,
    is_half,
    batched_infer_enabled
):
    global p_tts_inference
    if batched_infer_enabled:
        cmd = '"%s" -s GPT_SoVITS/inference_webui_fast.py "%s"' % (python_exec, language)
    else:
        cmd = '"%s" -s GPT_SoVITS/inference_webui.py "%s"' % (python_exec, language)
    # #####v3暂不支持加速推理
    # if version=="v3":
    #     cmd = '"%s" GPT_SoVITS/inference_webui.py "%s"'%(python_exec, language)
    if p_tts_inference is None:
        os.environ["gpt_path"] = gpt_path
        os.environ["sovits_path"] = sovits_path
        os.environ["sovits_v3_path"] = sovits_v3_path
        os.environ["sovits_v4_path"] = sovits_v4_path
        os.environ["cnhubert_base_path"] = cnhubert_base_path
        os.environ["bert_path"] = bert_path
        os.environ["bigvgan_path"] = bigvgan_path
        os.environ["vocoder_path"] = vocoder_path
        os.environ["sv_path"] = sv_path
        os.environ["g2pw_path"] = g2pw_path
        os.environ["_CUDA_VISIBLE_DEVICES"] = str(fix_gpu_number(gpu_number))
        os.environ["is_half"] = str(is_half)
        os.environ["infer_ttswebui"] = str(webui_port_infer_tts)
        os.environ["is_share"] = str(is_share)
        print("TTS推理进程已开启")
        print(cmd)
        p_tts_inference = Popen(cmd, shell=True, env=os.environ)
        p_tts_inference.wait()
    else:
        kill_process(p_tts_inference.pid, process_name_tts)
        p_tts_inference = None
        print("TTS推理进程已关闭")


def infer(
    version: str = "v4",
    sovits_path: str = "",
    sovits_v3_path: str = "",
    sovits_v4_path: str = "",
    gpt_path: str = "",
    cnhubert_base_path: str = "",
    bert_path: str = "",
    bigvgan_path: str = "",
    vocoder_path: str = "",
    sv_path: str = "",
    g2pw_path: str = "",
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
        sovits_v4_path = sovits_v4_path,
        bigvgan_path = bigvgan_path,
        vocoder_path = vocoder_path,
        sv_path = sv_path,
        g2pw_path = g2pw_path,
        is_half = half_precision,
        batched_infer_enabled = batched_infer,
    )

    # 2-GPT-SoVITS-变声