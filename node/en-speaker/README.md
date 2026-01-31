部署

    rsync -avP dist/ ihuhao:/home/ubuntu/src/html/en-speaker  

自建 hf 镜像

    pip install -U huggingface_hub
    export HF_ENDPOINT=https://hf-mirror.com
    hf download  onnx-community/Kokoro-82M-v1.0-ONNX --local-dir onnx-community/Kokoro-82M-v1.0-ONNX
    coscmd upload -rs --delete ./onnx-community /