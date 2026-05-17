FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    'transformers>=4.40' \
    accelerate \
    tqdm \
    huggingface_hub

COPY config/ config/
COPY src/ src/
COPY prompts/ prompts/

RUN mkdir -p data/stories logs

CMD ["python", "src/generate.py"]
