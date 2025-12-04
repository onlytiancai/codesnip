### 下载模型

    export HF_ENDPOINT=https://hf-mirror.com
    mkdir -p ~/models/Fara-7B
    huggingface-cli download microsoft/Fara-7B \
    --local-dir ~/models/Fara-7B \
    --local-dir-use-symlinks False \
    --resume-download

### 使用 vllm ，只能 cpu 推理，很慢，不可行

    export VLLM_USE_TORCH_COMPILE=0
    export PYTORCH_INDUCTOR=0

    vllm serve ~/models/Fara-7B \
    --port 5001 \
    --dtype auto \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --max_num_batched_tokens 4096


### 使用 llama.cpp

需要转换成gguf格式以及量化，但量化后不支持多模态

    cd ~/src/llama.cpp
    python3 convert_hf_to_gguf.py \
    ~/models/Fara-7B \
    --outfile ~/models/Fara-7B/Fara-7B-f16.gguf \
    --outtype f16

    ./build/bin/llama-quantize \
    ~/models/Fara-7B/Fara-7B-f16.gguf \
    ~/models/Fara-7B/Fara-7B-q4_K_M.gguf \
    Q4_K_M


    ./build/bin/llama-server \
    -m ~/models/Fara-7B/Fara-7B-q4_K_M.gguf \
    -c 2048 \
    -t 6 \
    -ngl 12 \
    --flash-attn 0 \
    --host 0.0.0.0 \
    --port 5001

### transformer

warmup 太慢了，半小时都没起来

安装依赖

    pip install fastapi uvicorn pillow transformers accelerate
    pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu

server.py

    import torch
    from fastapi import FastAPI, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
    from PIL import Image
    import io

    MODEL_PATH = "/Users/huhao/models/Fara-7B"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    )
    model.to(device)

    app = FastAPI()

    # 允许跨域（可选）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_headers=["*"],
        allow_methods=["*"],
    )


    @app.post("/v1/chat/completions")
    async def chat_completions(
        prompt: str,
        image: UploadFile | None = File(None),
    ):

        # --- 处理图像输入 ---
        if image is not None:
            img_bytes = await image.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            inputs = processor(images=img, text=prompt, return_tensors="pt")
        else:
            inputs = processor(text=prompt, return_tensors="pt")

        # 放到 MPS
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 推理
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )

        # 解码
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "model": "Fara-7B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result
                    }
                }
            ]
        }


    @app.get("/health")
    async def health():
        return {"status": "ok", "device": device}

启动

    uvicorn server:app --host 0.0.0.0 --port 5001

测试

    curl -X POST "http://localhost:5001/v1/chat/completions" \
        -F "prompt=你好，做一下自我介绍"

    curl -X POST "http://localhost:5001/v1/chat/completions" \
        -F "prompt=描述这张图片" \
        -F "image=@example.jpg"



### 使用

    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY no_proxy
    vi endpoint_configs/vllm_config.json # 5001
    fara-cli  --task "whats the weather in new york now" --endpoint_config endpoint_configs/vllm_config.json

### 量化版本

```
hf download bartowski/microsoft_Fara-7B-GGUF --include "microsoft_Fara-7B-Q4_K_M.gguf" --local-dir ./
~/src/llama.cpp/build/bin/llama-cli -m microsoft_Fara-7B-Q4_K_M.gguf -p "你好，帮我做个总结" --threads 8
~/src/llama.cpp/build/bin/llama-cli -m microsoft_Fara-7B-Q4_K_M.gguf --image ~/Pictures/logo128.png  -p "Please describe this image."   -n 512 -c 4096

hf download mradermacher/Fara-7B-GGUF --include "Fara-7B.mmproj-Q8_0.gguf" --local-dir ./
hf download mradermacher/Fara-7B-GGUF --include "Fara-7B.Q8_0.gguf" --local-dir ./
~/src/llama.cpp/build/bin/llama-mtmd-cli -m microsoft_Fara-7B-Q4_K_M.gguf --mmproj Fara-7B.mmproj-Q8_0.gguf --image ~/Pictures/logo128.png  -p "Please describe this image."   -n 512 -c 4096

~/src/llama.cpp/build/bin/llama-server \
    -m microsoft_Fara-7B-Q4_K_M.gguf \
    --mmproj Fara-7B.mmproj-Q8_0.gguf  \
    --ctx-size 16384 \
    --threads 8 \
    --host 0.0.0.0 \
    --port 8000

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Fara-7B",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "Describe this image." },
          {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,'$(base64 -i ~/Pictures/apollo.png)'"
            }
          }
        ]
      }
    ]
  }' | jq

cat endpoint_configs/vllm_config.json
{
    "model": "Fara-7B",
    "base_url": "http://localhost:8000/v1",
    "api_key": "<YOUR API KEY>"
}
fara-cli  --task "whats the weather in new york now" --endpoint_config endpoint_configs/vllm_config.json

```

### magentic-ui 

    pip install 'magentic-ui[fara]'

    cat fara_config.yaml
    model_config_local_surfer: &client_surfer
    provider: OpenAIChatCompletionClient
    config:
        model: "Fara-7B"
        base_url: http://localhost:8000/v1
        api_key: not-needed
        model_info:
        vision: true
        function_calling: true
        json_output: false
        family: "unknown"
        structured_output: false
        multiple_system_messages: false

    orchestrator_client: *client_surfer
    coder_client: *client_surfer
    web_surfer_client: *client_surfer
    file_surfer_client: *client_surfer
    action_guard_client: *client_surfer
    model_client: *client_surfer

    magentic-ui --fara --port 8081 --config fara_config.yaml


