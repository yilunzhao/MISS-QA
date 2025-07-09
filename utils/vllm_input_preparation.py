from transformers import AutoTokenizer
from utils.constant import COT_PROMPT
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser

from argparse import Namespace
from typing import List
import torch
from transformers import AutoProcessor, AutoTokenizer

from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import fetch_image
import os
import hashlib
import base64
import requests
from tqdm import tqdm
from utils.input_processing import prepare_qa_text_input, prepare_single_image_input

def prepare_qwen2_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA inputs for Qwen2-VL-7B-Instruct or similar models.
    """
    inputs = []
    llm = LLM(
        model=model_name,
        tensor_parallel_size=min(torch.cuda.device_count(), 4),
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1}  # Only allow 1 image
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=None
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256*28*28,
        max_pixels=1280*28*28
    )

    for query in tqdm(queries, desc="Prepare model inputs"):
        # 1. Prepare text QA prompt
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        
        # 2. Retrieve single base64 image (using your custom function)
        single_img_b64 = prepare_single_image_input(model_name, query['figure_path'])
        # 3. Convert base64 to an actual image object recognized by vLLM
        vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]

        # Build message structure for Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "dummy_path_for_vllm",  # placeholder
                    },
                    {"type": "text", "text": qa_text_prompt},
                ],
            }
        ]
        
        # Use Qwen's AutoProcessor to form a proper text input with generation prompt
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        item_input = {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)
    
    return inputs, llm, sampling_params


def prepare_phi3v_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA inputs for Microsoft Phi-3.5-vision-instruct or similar models.
    """
    inputs = []
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=min(torch.cuda.device_count(),4),
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=None
    )
    
    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # Convert base64 to actual image object
        single_img_b64 = prepare_single_image_input(model_name, query['figure_path'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]

        # Placeholders for Phi-3.5 format
        placeholders = "\n".join(f"<|image_{i}|>" for i, _ in enumerate(vision_input, start=1))
        text_input = f"<|user|>\n{placeholders}\n{qa_text_prompt}<|end|>\n<|assistant|>\n"
        
        item_input = {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)

    return inputs, llm, sampling_params

def prepare_phi4mm_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA inputs for Microsoft Phi-4
    """
    inputs = []
    llm = LLM(
            model=model_name,
            max_model_len=32768,
            trust_remote_code=True,
            max_num_batched_tokens=12800,
            enable_lora=True,
            max_lora_rank=320,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={"dynamic_hd": 16},
            tensor_parallel_size=min(torch.cuda.device_count(), 4)
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=None
    )
    
    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # Convert base64 to actual image object
        single_img_b64 = prepare_single_image_input(model_name, query['figure_path'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]

        # Placeholders for Phi-3.5 format
        placeholders = "\n".join(f"<|image_{i}|>" for i, _ in enumerate(vision_input, start=1))
        text_input = f"<|user|>\n{placeholders}\n{qa_text_prompt}<|end|>\n<|assistant|>\n"
        
        item_input = {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)

    return inputs, llm, sampling_params

def prepare_deepseek_vl2_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA for Deepseek-VL2 or similar models.
    """
    inputs = []
    llm = LLM(
        model=model_name,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=min(torch.cuda.device_count(),4),
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=None
    )
    
    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # Single image
        single_img_b64 = prepare_single_image_input(model_name, query['image'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]

        text_input = f"<|User|>: <image>\n{qa_text_prompt}\n\n<|Assistant|>:"
        
        item_input = {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)

    return inputs, llm, sampling_params

    
def prepare_llava_next_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    inputs = []
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=min(torch.cuda.device_count(),4),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # Single image
        single_img_b64 = prepare_single_image_input(model_name, query['image'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]

        text_input = f"[INST] <image>\n{qa_text_prompt} [/INST]"
        
        item_input = {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)

    return inputs, llm, sampling_params

def prepare_general_vlm_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA for a general VLM-based model (e.g., some HF large models).
    """
    inputs = []

    if "molmo" in model_name.lower():
        max_model_len = 4096
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 1},
            tensor_parallel_size=min(torch.cuda.device_count(),4),
        )
    else:
        max_model_len = 8192*4
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 1},
            tensor_parallel_size=min(torch.cuda.device_count(),4),
        )

    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # Single image
        single_img_b64 = prepare_single_image_input(model_name, query['figure_path'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]

        placeholders = "\n".join(f"Image-{i}: <image>\n"
                                 for i, _ in enumerate(vision_input, start=1))
        messages = [{'role': 'user', 'content': f"{placeholders}\n{qa_text_prompt}"}]
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        item_input = {
            "prompt": text_input,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)

    if "h2oai" in model_name:
        stop_token_ids = [tokenizer.eos_token_id]
    else:
        stop_token_ids = None

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids
    )

    return inputs, llm, sampling_params


def prepare_pixtral_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA for Pixtral models.
    """
    stop_token_ids = None
    llm = LLM(
        model=model_name,
        max_model_len=8192*4,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=min(torch.cuda.device_count(),4)
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids
    )
    
    inputs = []
    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        single_img_b64 = prepare_single_image_input(model_name, query['figure_path'])
        # For Pixtral, the original logic places "[IMG]" placeholders
        cur_input = f"<s>[INST]{qa_text_prompt}\n[IMG][/INST]"
        inputs.append(cur_input)
        
    return inputs, llm, sampling_params


def prepare_mllama_inputs(
    model_name,
    queries,
    prompt,
    temperature: float = 1,
    max_tokens: int = 1024
):
    """
    Prepare single-image QA for M-LLaMA or related models.
    """
    inputs = []
    llm = LLM(
        model=model_name,
        limit_mm_per_prompt={"image": 1},
        max_model_len=8192*4,
        max_num_seqs=2,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=min(torch.cuda.device_count(),4),
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=[128001,128008,128009]
    )

    for query in tqdm(queries, desc="Prepare model inputs"):
        # QA text
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        single_img_b64 = prepare_single_image_input(model_name, query['figure_path'])
        vision_input = [fetch_image(f"data:image/jpeg;base64,{single_img_b64}")]

        placeholders = "<|image|>"
        messages = [{'role': 'user', 'content': f"{placeholders}\n{qa_text_prompt}"}]
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        input_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        item_input = {
            "prompt": input_prompt,
            "multi_modal_data": {
                "image": vision_input
            },
        }
        inputs.append(item_input)

    return inputs, llm, sampling_params

