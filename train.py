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
        default_batch_size = int(minmem // 2 if version not in v3v4set else minmem // 8)
        default_batch_size_s1 = int(minmem // 2)
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


p_train_SoVITS = None
process_name_sovits = i18n("SoVITS训练")


def open1Ba(
    version,
    batch_size,
    total_epoch,
    exp_root,
    exp_name,
    exp_dir_weight,
    is_half,
    text_low_lr_rate,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers1Ba,
    if_grad_ckpt,
    lora_rank,
    pretrained_s2G,
    pretrained_s2D,
):
    global p_train_SoVITS
    if p_train_SoVITS == None:
        exp_name = exp_name.rstrip(" ")
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json"
        )
        with open(config_file) as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2_%s" % (s2_dir, version), exist_ok=True)
        # if check_for_existance([s2_dir], is_train=True):
        #     check_details([s2_dir], is_train=True)
        if is_half == False:
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["train"]["grad_ckpt"] = if_grad_ckpt
        data["train"]["lora_rank"] = lora_rank
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = exp_dir_weight
        data["name"] = exp_name
        data["version"] = version
        tmp_config_path = "%s/tmp_s2.json" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))
        if version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
            cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"' % (python_exec, tmp_config_path)
        else:
            cmd = '"%s" GPT_SoVITS/s2_train_v3_lora.py --config "%s"' % (python_exec, tmp_config_path)
        print("SoVITS训练开始：%s" % cmd)
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True, env=os.environ)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        print("SoVITS训练完成")
    else:
        print("已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务")


p_train_GPT = None
process_name_gpt = i18n("GPT训练")


def open1Bb(
    version,
    batch_size,
    total_epoch,
    exp_root,
    exp_name,
    exp_dir_weight,
    is_half,
    if_dpo,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers,
    pretrained_s1,
):
    global p_train_GPT
    if p_train_GPT == None:
        with open(
            "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml"
        ) as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        # if check_for_existance([s1_dir], is_train=True):
        #     check_details([s1_dir], is_train=True)
        if is_half == False:
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = exp_dir_weight
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1_%s" % (s1_dir, version)
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"] = str(fix_gpu_numbers(gpu_numbers.replace("-", ",")))
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" -s GPT_SoVITS/s1_train.py --config_file "%s" ' % (python_exec, tmp_config_path)
        print("GPT训练开始：%s"%cmd)
        p_train_GPT = Popen(cmd, shell=True, env=os.environ)
        p_train_GPT.wait()
        p_train_GPT = None
        print("GPT训练完成")
    else:
        print("已有正在进行的GPT训练任务，需先终止才能开启下一次任务")


ps1a = []
process_name_1a = i18n("文本分词与特征提取")


ps1b = []
process_name_1b = i18n("语音自监督特征提取")


ps1c = []
process_name_1c = i18n("语义Token提取")


ps1abc = []
process_name_1abc = i18n("训练集格式化一键三连")


def open1abc(
    version,
    inp_text,
    inp_wav_dir,
    exp_root,
    exp_name,
    is_half,
    gpu_numbers1a,
    gpu_numbers1Ba,
    gpu_numbers1c,
    g2pw_pretrained_dir,
    bert_pretrained_dir,
    ssl_pretrained_dir,
    sv_path,
    pretrained_s2G_path,
):
    global ps1abc
    exp_name = exp_name.rstrip(" ")
    if ps1abc == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        try:
            #############################1a
            path_text = "%s/2-name2text.txt" % opt_dir
            if os.path.exists(path_text) == False or (
                os.path.exists(path_text) == True
                and len(open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2
            ):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "g2pw_pretrained_dir": g2pw_pretrained_dir,
                    "bert_pretrained_dir": bert_pretrained_dir,
                    "is_half": str(is_half),
                }
                gpu_names = gpu_numbers1a.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets_1-get-text.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True, env=os.environ)
                    ps1abc.append(p)
                print("进度：1a-ing")
                for p in ps1abc:
                    p.wait()

                opt = []
                for i_part in range(all_parts):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                assert len("".join(opt)) > 0, process_info(process_name_1a, "failed")
            print("进度：1a-done")
            ps1abc = []
            #############################1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
                "sv_path": sv_path,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" -s GPT_SoVITS/prepare_datasets_2-get-hubert-wav32k.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True, env=os.environ)
                ps1abc.append(p)
            print("进度：1a-done, 1b-ing")
            for p in ps1abc:
                p.wait()
            ps1abc = []
            if "Pro" in version:
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets_2-get-sv.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True, env=os.environ)
                    ps1abc.append(p)
                for p in ps1abc:
                    p.wait()
                ps1abc = []
            print("进度：1a-done, 1b-done")
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if os.path.exists(path_semantic) == False or (
                os.path.exists(path_semantic) == True and os.path.getsize(path_semantic) < 31
            ):
                config_file = (
                    "GPT_SoVITS/configs/s2.json"
                    if version not in {"v2Pro", "v2ProPlus"}
                    else f"GPT_SoVITS/configs/s2{version}.json"
                )
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": config_file,
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets_3-get-semantic.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True, env=os.environ)
                    ps1abc.append(p)
                print("进度：1a1b-done, 1cing")
                for p in ps1abc:
                    p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                print("进度：all-done")
            ps1abc = []
            print("finish")
        except:
            traceback.print_exc()
            print("failed")
    else:
        print("occupy")


def train(
    version: str = "v4",
    fileList_path: str = "",
    modelPath_gpt: str = "",
    modelPath_sovitsG: str = "",
    modelPath_sovitsD: str = "",
    modelPath_sv: str = "",
    modelDir_bert: str = "",
    modelDir_hubert: str = "",
    modelDir_g2pw: str = "",
    half_precision: bool = False, # 16系卡没有半精度
    if_grad_ckpt: bool = False, # v3是否开启梯度检查点节省显存占用
    lora_rank: int = 32, # Lora秩 choices=[16, 32, 64, 128]
    output_root: str = "",
    output_dirName: str = "",
    output_logDir: str = "",
):
    os.makedirs(output_root, exist_ok = True)
    # To absolut audio path & get audio dir
    with open(file = fileList_path, mode = 'r', encoding = 'utf-8') as TextFile:
        Lines = TextFile.readlines()
    for Index, Line in enumerate(Lines):
        Line_Path, Line_SpeakerText = Line.split('|', maxsplit = 1)
        Line_Path = Path(fileList_path).parent.joinpath(Line_Path).as_posix()# if not Path(Line_Path).is_absolute() else Line_Path
        Line = f"{Line_Path}|{Line_SpeakerText}"
        Lines[Index] = Line
    fileList_path = Path(output_root).joinpath(Path(fileList_path).name).as_posix()
    with open(file = fileList_path, mode = 'w', encoding = 'utf-8') as TextFile:
        TextFile.writelines(Lines)
    Line_Path = Lines[0].split('|', maxsplit = 1)[0]
    assert Path(Line_Path).exists(), "请检查数据集是否为相对路径格式且音频在同一目录下"
    AudioDir = Path(Line_Path).parent.as_posix()
    # 
    os.environ["version"] = version
    set_default(version)
    # 1A-训练集格式化
    open1abc(
        version = version,
        inp_text = fileList_path,
        inp_wav_dir = AudioDir,
        exp_root = output_logDir,
        exp_name = output_dirName,
        is_half = half_precision,
        gpu_numbers1a = "%s-%s"%(gpus, gpus),
        gpu_numbers1Ba = "%s-%s"%(gpus, gpus),
        gpu_numbers1c = "%s-%s"%(gpus, gpus),
        g2pw_pretrained_dir = modelDir_g2pw,
        bert_pretrained_dir = modelDir_bert,
        ssl_pretrained_dir = modelDir_hubert,
        sv_path = modelPath_sv,
        pretrained_s2G_path = modelPath_sovitsG,
    )
    # 1B-SoVITS训练
    open1Ba(
        version = version,
        batch_size = default_batch_size,
        total_epoch = default_sovits_epoch,
        exp_root = output_logDir,
        exp_name = output_dirName,
        exp_dir_weight = output_root,
        is_half = half_precision,
        text_low_lr_rate = 0.4,
        if_save_latest = True,
        if_save_every_weights = True,
        save_every_epoch = default_sovits_save_every_epoch,
        gpu_numbers1Ba = "%s" % (gpus),
        if_grad_ckpt = if_grad_ckpt,
        lora_rank = int(lora_rank),
        pretrained_s2G = modelPath_sovitsG,
        pretrained_s2D = modelPath_sovitsD,
    )
    # 1B-GPT训练
    open1Bb(
        version = version,
        batch_size = default_batch_size_s1,
        total_epoch = 15,
        exp_root = output_logDir,
        exp_name = output_dirName,
        exp_dir_weight = output_root,
        is_half = half_precision,
        if_dpo = False,
        if_save_latest = True,
        if_save_every_weights = True,
        save_every_epoch = 5,
        gpu_numbers = "%s" % (gpus),
        pretrained_s1 = modelPath_gpt,
    )


if __name__ == '__main__':
    train(
        ...
    )