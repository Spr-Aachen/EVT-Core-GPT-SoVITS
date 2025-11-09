import argparse
import random
import uvicorn
from typing import Union, Optional
from fastapi import FastAPI
from PyEasyUtils import isPortAvailable, findAvailablePorts

from preprocess import Dataset_Creating
from train import train as _train
from infer import initialize as _initialize, handle as _handle, change_gpt_sovits_weights, handle_control as _handle_control, handle_change
from infer_webui import infer as _infer_webui


parser = argparse.ArgumentParser()
parser.add_argument("--host", help = "主机地址", type = str, default = "localhost")
parser.add_argument("--port", help = "端口",     type = int, default = 8080)
args = parser.parse_known_args()[0]

host = args.host
port = args.port if isPortAvailable(args.port, host) else random.choice(findAvailablePorts((8000, 8080)))


app = FastAPI()


@app.post("/terminate")
async def terminate():
    return _handle_control("exit")


@app.get("/createDataset")
async def createDataset(
    srtDir: str,
    audioSpeakersDataPath: str,
    # sampleRate: Optional[Union[int, str]] = 22050,
    # sampleWidth: Optional[Union[int, str]] = '32 (Float)',
    # toMono: bool = False,
    dataFormat: str = 'PATH|NAME|[LANG]TEXT[LANG]',
    trainRatio: float = 0.7,
    outputRoot: str = "./",
    outputDirName: str = "",
    fileListName: str = 'FileList',
):
    datasetCteator = Dataset_Creating(
        srtDir, audioSpeakersDataPath, dataFormat, trainRatio, outputRoot, outputDirName, fileListName
    )
    return datasetCteator.run()


@app.get("/train")
async def train(
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
    return _train(
        version, fileList_path, modelPath_gpt, modelPath_sovitsG, modelPath_sovitsD, modelPath_sv, modelDir_bert, modelDir_hubert, modelDir_g2pw, half_precision, if_grad_ckpt, lora_rank, output_root, output_dirName, output_logDir
    )


essentialsInitialized = False


@app.post("/tts_webui")
async def tts_webui(
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
    _infer_webui(
        version, sovits_path, sovits_v3_path, sovits_v4_path, gpt_path, cnhubert_base_path, bert_path, bigvgan_path, vocoder_path, sv_path, g2pw_path, half_precision, batched_infer
    )


@app.post("/tts_init")
async def tts_init(
    sovits_path: str,
    sovits_v3_path: str,
    sovits_v4_path: str,
    gpt_path: str,
    cnhubert_base_path: str,
    bert_path: str,
    bigvgan_path: str,
    vocoder_path: str = "",
    sv_path: str = "",
    g2pw_path: str = "",
    refer_wav_path: str = "", # 参考音频路径
    prompt_text: str = "", # 参考音频文本
    prompt_language: str = 'auto', # 参考音频语言 ['zh', 'yue', 'en', 'ja', 'ko', 'auto', 'auto_yue']
    device: str = 'cuda', # 生成引擎 ['cuda', 'cpu']
    half_precision: bool = True, # 是否使用半精度
    media_type: str = 'wav', # 音频格式 ['wav', 'ogg', 'aac']
    sub_type: str = 'int16', # 数据类型 ['int16', 'int32']
    stream_mode: str = 'normal', # 流式模式 ['close', 'normal', 'keepalive']
):
    global essentialsInitialized
    _initialize(
        sovits_path, sovits_v3_path, sovits_v4_path, gpt_path, refer_wav_path, prompt_text, prompt_language, device, half_precision, stream_mode, media_type, sub_type, cnhubert_base_path, bert_path, bigvgan_path, vocoder_path, sv_path, g2pw_path
    )
    essentialsInitialized = True


@app.get("/tts_handle")
async def tts_handle(
    refer_wav_path: str = "", # 参考音频路径
    prompt_text: str = "", # 参考音频文本
    prompt_language: str = 'auto', # 参考音频语言 ['zh', 'yue', 'en', 'ja', 'ko', 'auto', 'auto_yue']
    inp_refs: Optional[list] = None, # 辅助参考音频路径列表
    text: str = "", # 待合成文本
    text_language: str = 'auto', # 目标文本语言 ['zh', 'yue', 'en', 'ja', 'ko', 'auto', 'auto_yue']
    cut_punc: Optional[str] = None, # 文本切分符号 [',', '.', ';', '?', '!', '、', '，', '。', '？', '！', '；', '：', '…']
    top_k: int = 5, # Top-K 采样值
    top_p: float = 1.0, # Top-P 采样值
    temperature: float = 1.0, # 温度值
    speed: float = 1.0, # 语速因子
    sample_steps: int = 32, # 采样步数 [4, 8, 16, 32]
    if_sr: bool = False, # 是否超分
):
    global essentialsInitialized
    if not essentialsInitialized:
        raise Exception("Not initialized")
    return _handle(
        refer_wav_path, prompt_text, prompt_language, text, text_language, cut_punc, top_k, top_p, temperature, speed, inp_refs, sample_steps, if_sr
    )


if __name__ == "__main__":
    uvicorn.run(app, host = host, port = port)