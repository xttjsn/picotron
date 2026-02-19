FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY picotron.py train.py ./

CMD ["python", "train.py", "--steps", "200", "--device", "cuda", "--hidden", "256", "--layers", "4", "--heads", "4", "--seq-len", "128", "--micro-batch", "4"]
