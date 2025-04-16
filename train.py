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
    version,
):
    global p_train_SoVITS
    if p_train_SoVITS == None:
        with open("GPT_SoVITS/configs/s2.json") as f:
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
        if version in ["v1", "v2"]:
            cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"' % (python_exec, tmp_config_path)
        else:
            cmd = '"%s" GPT_SoVITS/s2_train_v3_lora.py --config "%s"' % (python_exec, tmp_config_path)
        # yield (
        #     process_info(process_name_sovits, "opened"),
        #     {"__type__": "update", "visible": False},
        #     {"__type__": "update", "visible": True},
        # )
        print("SoVITS训练开始：%s" % cmd)
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        # yield (
        #     process_info(process_name_sovits, "finish"),
        #     {"__type__": "update", "visible": True},
        #     {"__type__": "update", "visible": False},
        # )
        print("SoVITS训练完成")
    else:
        # yield (
        #     process_info(process_name_sovits, "occupy"),
        #     {"__type__": "update", "visible": False},
        #     {"__type__": "update", "visible": True},
        # )
        print("已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务")


p_train_GPT = None
process_name_gpt = i18n("GPT训练")


def open1Bb(
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
    version,
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

        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(gpu_numbers.replace("-", ","))
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" ' % (python_exec, tmp_config_path)
        # yield (
        #     process_info(process_name_gpt, "opened"),
        #     {"__type__": "update", "visible": False},
        #     {"__type__": "update", "visible": True},
        # )
        print("GPT训练开始：%s"%cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        # yield (
        #     process_info(process_name_gpt, "finish"),
        #     {"__type__": "update", "visible": True},
        #     {"__type__": "update", "visible": False},
        # )
        print("GPT训练完成")
    else:
        # yield (
        #     process_info(process_name_gpt, "occupy"),
        #     {"__type__": "update", "visible": False},
        #     {"__type__": "update", "visible": True},
        # )
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
    inp_text,
    inp_wav_dir,
    exp_root,
    exp_name,
    is_half,
    gpu_numbers1a,
    gpu_numbers1Ba,
    gpu_numbers1c,
    bert_pretrained_dir,
    ssl_pretrained_dir,
    pretrained_s2G_path,
):
    global ps1abc
    # inp_text = my_utils.clean_path(inp_text)
    # inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    # if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
    #     check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
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
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets_1-get-text.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                # yield (
                #     i18n("进度") + ": 1A-Doing",
                #     {"__type__": "update", "visible": False},
                #     {"__type__": "update", "visible": True},
                # )
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
            # yield (
            #     i18n("进度") + ": 1A-Done",
            #     {"__type__": "update", "visible": False},
            #     {"__type__": "update", "visible": True},
            # )
            print("进度：1a-done")
            ps1abc = []
            #############################1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets_2-get-hubert-wav32k.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            # yield (
            #     i18n("进度") + ": 1A-Done, 1B-Doing",
            #     {"__type__": "update", "visible": False},
            #     {"__type__": "update", "visible": True},
            # )
            print("进度：1a-done, 1b-ing")
            for p in ps1abc:
                p.wait()
            # yield (
            #     i18n("进度") + ": 1A-Done, 1B-Done",
            #     {"__type__": "update", "visible": False},
            #     {"__type__": "update", "visible": True},
            # )
            print("进度：all-done")
            ps1abc = []
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if os.path.exists(path_semantic) == False or (
                os.path.exists(path_semantic) == True and os.path.getsize(path_semantic) < 31
            ):
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": "GPT_SoVITS/configs/s2.json",
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets_3-get-semantic.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                # yield (
                #     i18n("进度") + ": 1A-Done, 1B-Done, 1C-Doing",
                #     {"__type__": "update", "visible": False},
                #     {"__type__": "update", "visible": True},
                # )
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
                # yield (
                #     i18n("进度") + ": 1A-Done, 1B-Done, 1C-Done",
                #     {"__type__": "update", "visible": False},
                #     {"__type__": "update", "visible": True},
                # )
                print("进度：all-done")
            ps1abc = []
            # yield (
            #     process_info(process_name_1abc, "finish"),
            #     {"__type__": "update", "visible": True},
            #     {"__type__": "update", "visible": False},
            # )
            print("一键三连进程结束")
        except:
            traceback.print_exc()
            # yield (
            #     process_info(process_name_1abc, "failed"),
            #     {"__type__": "update", "visible": True},
            #     {"__type__": "update", "visible": False},
            # )
            print("一键三连中途报错")
    else:
        # yield (
        #     process_info(process_name_1abc, "occupy"),
        #     {"__type__": "update", "visible": False},
        #     {"__type__": "update", "visible": True},
        # )
        print("已有正在进行的一键三连任务，需先终止才能开启下一次任务")


def train(
    version: str = "v3",
    fileList_path: str = "GPT-SoVITS/raw/xxx.list",
    modelDir_bert: str = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    modelDir_hubert: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    modelPath_gpt: str = "GPT_SoVITS/pretrained_models/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    modelPath_sovitsG: str = "GPT_SoVITS/pretrained_models/s2G2333k.pth",
    modelPath_sovitsD: str = "GPT_SoVITS/pretrained_models/s2D2333k.pth",
    #set_gpt_batchSize: int = default_batch_size_s1,
    #set_gpt_epochs: int = ,
    #set_gpt_saveInterval: int = ,
    #set_sovits_batchSize: int = default_batch_size,
    #set_sovits_epochs: int = default_sovits_epoch,
    #set_sovits_saveInterval: int = default_sovits_save_every_epoch,
    half_precision: bool = False, # 16系卡没有半精度
    if_grad_ckpt: bool = False, # v3是否开启梯度检查点节省显存占用
    lora_rank: int = 32, # Lora秩 choices=[16, 32, 64, 128]
    output_root: str = "SoVITS_weights&GPT_weights",
    output_dirName: str = "模型名",
    output_logDir: str = "logs",
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
        inp_text = fileList_path,
        inp_wav_dir = AudioDir,
        exp_root = output_logDir,
        exp_name = output_dirName,
        is_half = half_precision,
        gpu_numbers1a = "%s-%s"%(gpus, gpus),
        gpu_numbers1Ba = "%s-%s"%(gpus, gpus),
        gpu_numbers1c = "%s-%s"%(gpus, gpus),
        bert_pretrained_dir = modelDir_bert,
        ssl_pretrained_dir = modelDir_hubert,
        pretrained_s2G_path = modelPath_sovitsG
    )
    # 1B-SoVITS训练
    open1Ba(
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
        version = version
    )
    # 1B-GPT训练
    open1Bb(
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
        version = version
    )


if __name__ == '__main__':
    train(
        'v2',
        'd:/Projekt/Git/Python Projects/EVT - test/数据集制作结果/GPT-SoVITS/4/Train_2025-04-06.txt',
        'd:/Projekt/Git/Python Projects/EVT - test/Models/TTS/GPT-SoVITS/Downloaded/chinese-roberta-wwm-ext-large',
        'd:/Projekt/Git/Python Projects/EVT - test/Models/TTS/GPT-SoVITS/Downloaded/chinese-hubert-base',
        'd:/Projekt/Git/Python Projects/EVT - test/Models/TTS/GPT-SoVITS/Downloaded/s1&s2/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt',
        'd:/Projekt/Git/Python Projects/EVT - test/Models/TTS/GPT-SoVITS/Downloaded/s1&s2/s2G2333k.pth',
        'd:/Projekt/Git/Python Projects/EVT - test/Models/TTS/GPT-SoVITS/Downloaded/s1&s2/s2D2333k.pth',
        half_precision = True,
        if_grad_ckpt = False,
        lora_rank = '32',
        output_root = 'd:/Projekt/Git/Python Projects/EVT - test/模型训练结果/GPT-SoVITS',
        output_dirName = '4',
        output_logDir = '/EVT_TrainLog/GPT-SoVITS/2025-04-06'
    )