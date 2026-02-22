# Use a base image
#FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
#FROM 12.9.0-cudnn-devel-ubuntu24.04
#FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04
#FROM nvidia/cuda:13.0.1-base-ubuntu24.04
#FROM nvcr.io/nvidia/pytorch:25.12-py3

# Set environment variables
#ENV MY_VAR="Hello, World!"

# Install dependencies
RUN apt update && apt install -y --no-install-recommends python3 python-is-python3 python3-pip python3-venv python3-dev

# Setting up python
WORKDIR /setup
COPY requirements.txt /setup/
RUN python3 -m venv /setup/venv
RUN /bin/bash -c "source /setup/venv/bin/activate && python -m pip install --no-cache-dir -r requirements.txt"

# Set the command to run
WORKDIR /work-dir
CMD [ "/bin/bash", "-c", "source /setup/venv/bin/activate && exec bash" ]
