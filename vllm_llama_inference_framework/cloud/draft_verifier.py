"""
F2: 云端 Draft 验证器 - 鲁棒匹配修复版
使用最长公共前缀 (LCP) 算法，彻底解决 split(' ') 带来的错位问题
"""
import asyncio
import time
from typing import List, Tuple, Dict, Any, Optional

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("❌ 警告: 未找到 vLLM")
    LLM = Any 
    SamplingParams = Any

from common.types import VerifyRequest, VerifyResponse

class DraftVerifier:
    def __init__(self, model_path: str, acceptance_threshold: float = 0.8):
        self.model_path = model_path
        self.acceptance_threshold = acceptance_threshold
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        print(f"[Cloud] 加载 vLLM 模型: {model_path} (TP自动适配)")
        import torch
        gpu_count = torch.cuda.device_count()
        tp_size = 4 if gpu_count >= 4 else 1
        
        try:
            return LLM(
                model=model_path,
                tensor_parallel_size=tp_size,
                dtype="float16",
                trust_remote_code=True,
                gpu_memory_utilization=0.85,
                max_model_len=2048,
                enforce_eager=False
            )
        except Exception as e:
            print(f"❌ vLLM 初始化失败: {e}")
            raise e

    async def verify_draft(self, request: VerifyRequest) -> VerifyResponse:
        """验证 Draft (字符串级精准匹配)"""
        start_time = time.time()
        
        full_prompt = request.prompt
        # 1. 还原端侧生成的完整字符串
        draft_text_raw = "".join(request.draft_tokens)
        
        # 2. 让云端生成标准答案 (Ground Truth)
        # 长度只要比 draft 稍微长一点即可，确保能覆盖
        max_verify_len = len(request.draft_tokens) + 20
        
        cloud_generated_text = await self._generate_ground_truth(
            full_prompt, 
            max_tokens=max_verify_len
        )
        
        # 3. 🚀 核心逻辑: 最长公共前缀匹配 (Character-level LCP)
        match_len = 0
        min_len = min(len(draft_text_raw), len(cloud_generated_text))
        
        # 逐字符比对
        for i in range(min_len):
            if draft_text_raw[i] == cloud_generated_text[i]:
                match_len += 1
            else:
                break
        
        # 4. 判断结果
        # accepted_text 是 draft 中匹配成功的部分
        accepted_text = draft_text_raw[:match_len]
        # rejected_text 是 draft 中错误的部分
        rejected_text = draft_text_raw[match_len:]
        
        # 计算 Token 级的接受率 (估算)
        # 我们用字符长度比例来估算，或者简单地看 draft 是否被完全接受
        is_fully_accepted = (match_len == len(draft_text_raw))
        
        # 统计 "被修正的 Token 数"
        # 这是一个近似值，因为我们现在是字符级比对。
        # 逻辑：如果 draft 长度是 100 字符，匹配了 80 字符，那我们就认为 20% 的 token 错了。
        total_chars = len(draft_text_raw)
        if total_chars > 0:
            acceptance_rate = match_len / total_chars
        else:
            acceptance_rate = 1.0 # 空草稿算全对
            
        print(f"[Cloud] 验证结果: Draft长={len(draft_text_raw)}, 匹配长={match_len}, 接受率={acceptance_rate:.1%}")
        
        # 5. 构造最终输出
        # 最终文本 = Prompt + (匹配的 Draft 部分) + (Cloud 生成的剩余部分)
        # Cloud 生成的剩余部分 = cloud_generated_text[match_len:]
        correction = cloud_generated_text[match_len:]
        final_text = accepted_text + correction
        
        # 为了兼容接口返回 tokens 列表，我们简单切分一下 (仅用于显示)
        # 注意：这里的 tokens 并不严格对应模型 tokenizer，仅供前端或日志查看
        verified_tokens = [accepted_text, correction] 
        
        # 修正位置：这里不再返回具体的 token index 列表，因为字符级无法精确对应 token index
        # 只要 acceptance_rate < 1.0，就说明末尾有修正
        corrected_positions = [-1] if not is_fully_accepted else []

        latency = (time.time() - start_time) * 1000
        
        return VerifyResponse(
            verified_tokens=verified_tokens,
            verified_token_ids=[],
            accepted_count=match_len, # 这里借用字段存字符数
            total_count=total_chars,  # 这里借用字段存字符数
            acceptance_rate=acceptance_rate,
            corrected_positions=corrected_positions,
            final_text=full_prompt + final_text, # 返回包含 prompt 的全量文本
            latency_ms=latency
        )
    
    async def _generate_ground_truth(self, prompt: str, max_tokens: int) -> str:
        """调用 vLLM 生成"""
        sampling_params = SamplingParams(
            temperature=0.0, # 验证必须用贪心
            max_tokens=max_tokens
        )
        
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None, 
            lambda: self.model.generate([prompt], sampling_params)
        )
        return output[0].outputs[0].text