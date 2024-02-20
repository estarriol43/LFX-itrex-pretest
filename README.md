# LFX-itrex

## Hardware Environment
Note: I used the dockerfile below for the environment setup
![image](https://hackmd.io/_uploads/HyKWE9b2T.png)


## Part One: Intel extension for transformers framework

### Environment setup
I followed the instructions [here](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/installation.md) and [here](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/installation.md) to set up my environment. I also installed some missing packages manually based on the error messages. The following is the dockerfile I used.

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libsm6 \
    libxext6 \
    git \
    python3-pip \
    python-is-python3 \
    software-properties-common \
    cmake \
    libboost-all-dev \
    llvm-14-dev \
    liblld-14-dev \
    gcc \
    g++ \
    clang-14 \
    ninja-build \
    curl

RUN git clone https://github.com/intel/neural-speed.git && \
    cd neural-speed && \
    pip install -r requirements.txt && \
    pip install .

RUN pip install intel-extension-for-transformers && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install intel-extension-for-pytorch && \
    pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```

To build and run the dockerfile, I used the below command:
```bash
docker build -t ubuntu-itrex .
docker run -it -p 8080:8080 -v [WORKING_DIR]:/lfx-wasmedge ubuntu-itrex bash
```

- `WORKING_DIR` is a shared folder between the host and the docker environment. I put the required source code and the model in this folder
- `-p 8080:8080` is the port forwarding option, which would be used for the API server example.

### Transformers-based extension APIs

- Get model

    [git lfs](https://github.com/git-lfs/git-lfs) should be installed first. I used [Intel/neural-chat-7b-v3-1](https://huggingface.co/Intel/neural-chat-7b-v3-1) for example.
```bash
git lfs install
git clone https://huggingface.co/Intel/neural-chat-7b-v3-1
```

- Source code

    I added a streamer into the sample code, so we can see the output generated word by word. Besides, I also restricted the maximum number of token generated to 100.
```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "neural-chat-7b-v3-1"     # Hugging Face model_id or local model
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=100)
```

### Result
![image](https://hackmd.io/_uploads/Sy9z5qxh6.png)

## Part Two: llama.cpp and chatbot/API sever

### Environment setup
I re-used the Dockerfile in part one.

### Build and Install llama.cpp
I followed the instruction [here](https://wasmedge.org/docs/contribute/source/plugin/wasi_nn/#build-wasmedge-with-wasi-nn-llamacpp-backend) to build and install `llama.cpp` plugin
```bash
git clone https://github.com/WasmEdge/WasmEdge.git
git checkout origin/hydai/0.13.5_ggml_lts

cd WasmEdge

cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release \
  -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="GGML" \
  -DWASMEDGE_PLUGIN_WASI_NN_GGML_LLAMA_BLAS=OFF \
  .

cmake --build build

# For the WASI-NN plugin, you should install this project.
cmake --install build
```
### Chat Example
I followed the instruction [here](https://github.com/second-state/WasmEdge-WASINN-examples/tree/master/wasmedge-ggml#llama-example-for-wasi-nn-with-ggml-backend) to execute the chatbot example

- Build and install WASI-NN ggml plugin
```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

source $HOME/.bashrc
```
- Get the model and sample code
```bash
curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

git clone https://github.com/second-state/WasmEdge-WASINN-examples.git
```

- Execute
```bash
cp WasmEdge-WASINN-examples/wasmedge-ggml/llama-stream/wasmedge-ggml-llama-stream.wasm ./

wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-llama-stream.wasm default
```

- Result
![image](https://hackmd.io/_uploads/rJtnkgb2p.png)

### API server example
I followed the instruction [here](https://github.com/second-state/LlamaEdge/tree/main/api-server#create-an-openai-compatible-api-server-for-your-llm) to execute the API server example

- Build and install WASI-NN ggml plugin
```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml

source $HOME/.bashrc
```

- Get the model and the source code
```bash
# Get model
curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
# Get source code
curl -LO https://github.com/second-state/LlamaEdge/releases/latest/download/llama-api-server.wasm

# For WebUI
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
```

- Execute
```
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf llama-api-server.wasm -p llama-2-chat
```

- Test via RESTful
```bash
curl -X POST http://localhost:8080/v1/chat/completions -H 'accept:application/json' -H 'Content-Type: application/json' -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"llama-2-chat"}'
```

- Result of RESTful API
![image](https://hackmd.io/_uploads/rklMblW26.png)

- Result of WebUI
![image](https://hackmd.io/_uploads/Hkk9mcb3p.png)
