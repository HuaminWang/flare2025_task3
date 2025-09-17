# FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
# FROM pytorch/pytorch:1.12.0-cuda11.6-cudnn8-runtime
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
# FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime
WORKDIR /workspace
COPY . /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install -r /workspace/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN chmod +x /workspace/predict.sh

CMD ["/bin/bash"]


