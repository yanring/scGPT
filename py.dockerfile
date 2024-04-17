    FROM nvcr.io/nvidia/pytorch:24.03-py3 as base
    
    ENV SHELL /bin/bash
    RUN apt-get update
    RUN apt-get install -y sudo gdb pstack bash-builtins git zsh autojump tmux git curl 
    RUN pip install debugpy dm-tree torch_tb_profiler einops wandb
    RUN pip install sentencepiece tokenizers transformers torchvision ftfy modelcards datasets tqdm pydantic==2.2.1
    RUN pip install torch_geometric cell-gears==0.0.1 "flash-attn<1.0.5" dcor
    # RUN pip install nvidia-pytriton pylint py-spy yapf darker
    # RUN TORCH_CUDA_ARCH_LIST="8.0 9.0" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.0

    FROM base as dev
    RUN addgroup --gid 44408 zijiey
    RUN groupmod --gid 3000 dip
    RUN addgroup --gid 30 hardware
    RUN adduser --disabled-password --gecos GECOS -u 44408 -gid 30 zijiey
    RUN adduser zijiey sudo
    RUN usermod -a -G hardware zijiey
    RUN usermod -a -G hardware root
    RUN usermod -a -G root zijiey
    RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
    USER zijiey
    
