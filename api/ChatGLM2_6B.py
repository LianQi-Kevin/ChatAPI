"""
* ChatGLM2-6B GitHub: https://github.com/THUDM/ChatGLM2-6B
* fastllm GitHub: https://github.com/ztxz16/fastllm

example:
    from fastllm_pytools import llm
    model = llm.from_hf(model, tokenizer, dtype=quantize)  # dtype支持 "float16", "int8", "int4"
"""
import json
import logging
import time
from typing import List, Tuple, Literal, Optional

import torch
from flask import Blueprint, jsonify, request
from pydantic import BaseModel, ValidationError, Field, AliasChoices
from transformers import AutoTokenizer, AutoModel

from utils.api_utils import RESTfulAPI

ChatGLM2 = Blueprint('ChatGLM2', __name__, url_prefix="/ChatGLM2")

# Global Config
with open("./config.json") as f:
    MODEL_CONFIGS: dict = json.loads("".join(f.readlines()))["model"]

# CUDA DEVICE
DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# MODEL
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_CONFIGS["tokenizer_path"],
        revision=MODEL_CONFIGS["revision"],
        trust_remote_code=True,
        device=DEVICE
    )
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=MODEL_CONFIGS["model_path"],
        revision=MODEL_CONFIGS["revision"],
        trust_remote_code=True,
        device=DEVICE
    )
    if MODEL_CONFIGS["quantize"] != "float16":
        model.quantize(8 if MODEL_CONFIGS["quantize"] == "int8" else 4)  # quantize
    model.eval()
    return model, tokenizer


MODEL, TOKENIZER = get_model()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequestFormat(BaseModel):
    """转换请求到标准序列"""
    model: str = Field(..., validation_alias=AliasChoices("model_name", "model"))
    messages: List[ChatMessage] = Field(..., validation_alias=AliasChoices("conversations", "messages"))
    temperature: float = MODEL_CONFIGS.get("temperature", 0.7)
    top_p: float = MODEL_CONFIGS.get("top_p", 0.95)
    max_length: int = Field(default=MODEL_CONFIGS.get("max_length", 2048),
                            validation_alias=AliasChoices("max_tokens", "max_length"))


class ChatResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatResponseUsage(BaseModel):
    prompt_tokens: Optional[int] = Field(gt=-1)
    completion_tokens: Optional[int] = Field(gt=-1)
    total_tokens: Optional[int] = Field(gt=-1)


class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatResponseChoice]
    usage: ChatResponseUsage


def ChatGLM_format(messages: List[ChatMessage]) -> Tuple[str, List[tuple]]:
    """校验并提取history和prompt"""
    if messages[-1].role != "user":
        raise ValueError("Messages Error")
    else:
        prompt = messages[-1].content
    messages = messages[1: -1]  # remove system msg and prompt
    history = [(messages[i].content, messages[i + 1].content) for i in range(0, len(messages), 2) if
               messages[i].role == "user" and messages[i + 1].role == "assistant"] if len(messages) % 2 == 0 else []
    return prompt, history


def get_tokens(history: List[Tuple[str]]) -> int:
    """获取history的token数"""
    total_tokens = 0
    if len(history) != 0:
        for item in history:
            total_tokens += len(TOKENIZER(f"{item[0]} {item[1]}", add_special_tokens=True)['input_ids'])
    return total_tokens


def token_del_conversation(history: List[Tuple[str]], max_length: int = 250, token_limit: int = 4096) -> List[
    Tuple[str]]:
    """获取对话总token，从头删除超出token的数据(规避system信息)"""
    conv_history_tokens = get_tokens(history)
    while conv_history_tokens + max_length >= token_limit:
        del history[0]
        conv_history_tokens = get_tokens(history)
    return history


def torch_gc():
    """Collects GPU Memory"""
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


@ChatGLM2.route('/name', methods=['GET'])
def model_name():
    global MODEL_CONFIGS
    return jsonify(RESTfulAPI(code=200, status="success", data=MODEL_CONFIGS["name"]).model_dump()), 200


@ChatGLM2.route('/chat', methods=["POST"])
def chat():
    global MODEL, TOKENIZER
    try:
        params = ChatRequestFormat(**request.json)
        prompt, history = ChatGLM_format(messages=params.messages)
        history = token_del_conversation(history=history, max_length=params.max_length,
                                         token_limit=MODEL_CONFIGS.get("token_limit", 4096))
        response, _ = MODEL.chat(
            tokenizer=TOKENIZER,
            query=prompt,
            history=history,
            max_length=params.max_length,
            top_p=params.top_p,
            temperature=params.temperature,
        )
        # get tokens
        prompt_tokens = get_tokens(history) + len(TOKENIZER(f"{prompt}", add_special_tokens=True)['input_ids'])
        completion_tokens = len(TOKENIZER(f"{response}", add_special_tokens=True)['input_ids'])

        # create response
        api_response = RESTfulAPI(code=201, status="success",
                                  data=ChatResponse(
                                      id=MODEL_CONFIGS.get("name", "ChatGLM2-6B"),
                                      object="chat.completion",
                                      choices=[{
                                          "index": 0,
                                          "message": {"role": "assistant", "content": response},
                                          "finish_reason": "stop"
                                      }],
                                      usage={
                                          "prompt_tokens": prompt_tokens,
                                          "completion_tokens": completion_tokens,
                                          "total_tokens": prompt_tokens + completion_tokens
                                      })).model_dump()
        return jsonify(api_response), 201
    except (ValidationError, ValueError) as e:
        print(e)
        return jsonify(RESTfulAPI(code=400, status="error", message=f"{e}").model_dump()), 400
    except RuntimeError as _:
        try:
            logging.error(f"CUDA out of Memory, Restarting")
            return jsonify(RESTfulAPI(code=500, status="error",
                                      message="CUDA out of Memory, Restarting now, Please Try again").model_dump()), 500
        finally:
            # Reloading
            del MODEL
            del TOKENIZER
            MODEL, TOKENIZER = get_model()
    except Exception as e:
        return jsonify(RESTfulAPI(code=500, status="error", message=f"{e}").model_dump()), 500
    finally:
        torch_gc()
