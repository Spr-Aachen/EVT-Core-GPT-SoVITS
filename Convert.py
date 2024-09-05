import os,sys
if len(sys.argv)==1:sys.argv.append('v2')
version="v1"if sys.argv[1]=="v1" else"v2"
os.environ["version"]=version
import warnings
warnings.filterwarnings("ignore")
import torch
import platform
import psutil
import signal
import subprocess
from scipy.io.wavfile import write

from pathlib import Path
current_dir = Path(__file__).absolute().parent.as_posix()
sys.path.insert(0, f"{current_dir}")
os.chdir(current_dir)

from config import python_exec, webui_port_infer_tts, is_share


current_dir = Path(__file__).absolute().parent.as_posix()
os.chdir(current_dir)
sys.path.insert(0, f"{current_dir}/GPT_SoVITS")


def RunCMD(Args: str):
    Encoding = 'gbk' if platform.system() == 'Windows' else 'utf-8'

    TotalInput = f"{Args}\n".encode(Encoding)
    if platform.system() == 'Windows':
        ShellArgs = ['cmd']
    if platform.system() == 'Linux':
        ShellArgs = ['bash', '-c']
    Subprocess = subprocess.Popen(
        args = ShellArgs,
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT
    )
    TotalOutput = Subprocess.communicate(TotalInput)[0]
    TotalOutput = '' if TotalOutput is None else TotalOutput.strip().decode(Encoding)

    return None if TotalOutput == '' else TotalOutput


NetStat = RunCMD(f'netstat -aon|findstr "{webui_port_infer_tts}"')
for Line in str(NetStat).splitlines():
    Line = Line.strip()
    RunCMD(f'taskkill /PID {Line.split(" ")[-1]}') if Line.startswith("TCP") else None


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


def kill_process(pid):
    if(platform.system()=="Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)

           
gpu_infos = []
mem = []
if_gpu_ok = False

# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
ok_gpu_keywords={"10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L4","4060","H"}
set_gpu_numbers=set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))
'''
# 判断是否支持mps加速
if torch.backends.mps.is_available():
    if_gpu_ok = True
    gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
    mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # 实测使用系统内存作为显存不会爆显存
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # 当遇到mps不支持的步骤时使用cpu
'''

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("%s\t%s" % ("0", "CPU"))
    gpu_infos.append("%s\t%s" % ("0", "CPU"))
    set_gpu_numbers.add(0)
    default_batch_size = int(psutil.virtual_memory().total/ 1024 / 1024 / 1024 / 2)
gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers=str(sorted(list(set_gpu_numbers))[0])
def fix_gpu_number(input):#将越界的number强制改到界内
    try:
        if(int(input)not in set_gpu_numbers):return default_gpu_numbers
    except:return input
    return input
def fix_gpu_numbers(inputs):
    output=[]
    try:
        for input in inputs.split(","):output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


p_tts_inference=None
def change_tts_inference(
    if_tts,
    bert_path,
    cnhubert_base_path,
    gpu_number,
    is_half,
    gpt_path,
    sovits_path,
    batched_infer_enabled,
    use_webui
):
    global p_tts_inference
    if(if_tts==True and p_tts_inference==None):
        os.environ["gpt_path"]=gpt_path #if "/" in gpt_path else "%s/%s"%(GPT_weight_root,gpt_path)
        os.environ["sovits_path"]=sovits_path #if "/"in sovits_path else "%s/%s"%(SoVITS_weight_root,sovits_path)
        os.environ["cnhubert_base_path"]=cnhubert_base_path
        os.environ["bert_path"]=bert_path
        os.environ["_CUDA_VISIBLE_DEVICES"]=fix_gpu_number(gpu_number)
        os.environ["is_half"]=str(is_half)
        os.environ["infer_ttswebui"]=str(webui_port_infer_tts)
        os.environ["is_share"]=str(is_share)
        print("TTS推理进程已开启")
        if use_webui:
            cmd = '"%s" GPT_SoVITS/inference_webui_fast.py'%(python_exec) if batched_infer_enabled else '"%s" GPT_SoVITS/inference_webui.py'%(python_exec)
        else:
            cmd = f'"{python_exec}" "GPT_SoVITS/inference_gui.py"'
        #print('webui_port:', webui_port_infer_tts)
        p_tts_inference = subprocess.Popen(cmd, shell=True)
        p_tts_inference.wait()
    elif(if_tts==False and p_tts_inference!=None):
        kill_process(p_tts_inference.pid)
        p_tts_inference=None
        print("TTS推理进程已关闭")


def Convert(
    Model_Path_Load_s1: str = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    Model_Path_Load_s2G: str = "GPT_SoVITS/pretrained_models/s2G488k.pth",
    Model_Dir_Load_bert: str = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    Model_Dir_Load_ssl: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    Ref_Audio: str = "",
    Ref_Text_Free: bool = False,
    Ref_Text: str = "",
    Ref_Language: str = "多语种混合",
    Text: str = '请输入语句',
    Language: str = "多语种混合",
    How_To_Cut: str = "按标点符号切",
    Top_K: int = 5,
    Top_P: float = 1.,
    Temperature: float = 1.,
    Set_FP16_Run: bool = False,
    Audio_Path_Save: str = ...,
    Enable_Batched_Infer: bool = False,
    Use_WebUI: bool = False
):
    # 1C-推理
    change_tts_inference(
        if_tts = True,
        bert_path = Model_Dir_Load_bert,
        cnhubert_base_path = Model_Dir_Load_ssl,
        gpu_number = gpus,
        is_half = Set_FP16_Run,
        gpt_path = Model_Path_Load_s1,
        sovits_path = Model_Path_Load_s2G,
        batched_infer_enabled = Enable_Batched_Infer,
        use_webui = Use_WebUI
    )

    # 2-GPT-SoVITS-变声