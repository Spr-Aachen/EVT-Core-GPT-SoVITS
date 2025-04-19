from typing import Optional
from fastapi import Query

from convert import initialize, handle, change_gpt_sovits_weights, handle_control, handle_change


class TTSManager:
    """
    """
    def __init__(self):
        self.essentialsInitialized = False

    async def control(self,
        command: str = None
    ):
        return handle_control(command)

    async def tts_init(self,
        sovits_path: str,
        sovits_v3_path: str,
        gpt_path: str,
        cnhubert_base_path: str,
        bert_path: str,
        bigvgan_path: str,
        refer_wav_path: str = ..., # 参考音频路径
        prompt_text: str = ..., # 参考音频文本
        prompt_language: str = 'auto', # 参考音频语言 ['zh', 'yue', 'en', 'ja', 'ko', 'auto', 'auto_yue']
        device: str = 'cuda', # 生成引擎 ['cuda', 'cpu']
        half_precision: bool = True, # 是否使用半精度
        media_type: str = 'wav', # 音频格式 ['wav', 'ogg', 'aac']
        sub_type: str = 'int16', # 数据类型 ['int16', 'int32']
        stream_mode: str = 'normal', # 流式模式 ['close', 'normal', 'keepalive']
    ):
        initialize(sovits_path, sovits_v3_path, gpt_path, refer_wav_path, prompt_text, prompt_language, device, half_precision, stream_mode, media_type, sub_type, cnhubert_base_path, bert_path, bigvgan_path)
        self.essentialsInitialized = True

    async def tts_handle(self,
        refer_wav_path: str = ..., # 参考音频路径
        prompt_text: str = ..., # 参考音频文本
        prompt_language: str = 'auto', # 参考音频语言 ['zh', 'yue', 'en', 'ja', 'ko', 'auto', 'auto_yue']
        inp_refs: Optional[list] = None, # 辅助参考音频路径列表
        text: str = ..., # 待合成文本
        text_language: str = 'auto', # 目标文本语言 ['zh', 'yue', 'en', 'ja', 'ko', 'auto', 'auto_yue']
        cut_punc: Optional[str] = None, # 文本切分符号 [',', '.', ';', '?', '!', '、', '，', '。', '？', '！', '；', '：', '…']
        top_k: int = 5, # Top-K 采样值
        top_p: float = 1.0, # Top-P 采样值
        temperature: float = 1.0, # 温度值
        speed: float = 1.0, # 语速因子
        sample_steps: int = 32, # 采样步数 [4, 8, 16, 32]
        if_sr: bool = False, # 是否超分
    ):
        if not self.essentialsInitialized:
            raise Exception("Not initialized")
        return handle(refer_wav_path, prompt_text, prompt_language, text, text_language, cut_punc, top_k, top_p, temperature, speed, inp_refs, sample_steps, if_sr)