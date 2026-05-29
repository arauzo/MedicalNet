# podman run -d   -v /home/arauzo/MedicalNet/:/work-dir   --gpus=all   --shm-size=2g   --name medicalnet-train-spine-simple \
#   medical-net-devel bash -lc "source /setup/venv/bin/activate && \
#   python train_classification.py data/spine/ sin_fractura.txt con_fractura.txt -m simple -e 300"
# podman wait medicalnet-train-spine-simple

podman run -d   -v /home/arauzo/MedicalNet/:/work-dir   --gpus=all   --shm-size=2g   --name medicalnet-train-spine-resnet50 \
  medical-net-devel bash -lc "source /setup/venv/bin/activate && \
  python train_classification.py data/spine/ sin_fractura.txt con_fractura.txt -m resnet50 -e 300"
podman wait medicalnet-train-spine-resnet50

podman run -d   -v /home/arauzo/MedicalNet/:/work-dir   --gpus=all   --shm-size=2g   --name medicalnet-train-spine-resnet50-f \
  medical-net-devel bash -lc "source /setup/venv/bin/activate && \
  python train_classification.py data/spine/ sin_fractura.txt con_fractura.txt -m resnet50 -f -e 300"
podman wait medicalnet-train-spine-resnet50-f

# podman run -d   -v /home/arauzo/MedicalNet/:/work-dir   --gpus=all   --shm-size=2g   --name medicalnet-train-cuerpoV-simple \
#   medical-net-devel bash -lc "source /setup/venv/bin/activate && \
#   python train_classification.py data/cuerpoV/ sin_fractura.txt con_fractura.txt -m simple -e 300"
# podman wait medicalnet-train-cuerpoV-simple

podman run -d   -v /home/arauzo/MedicalNet/:/work-dir   --gpus=all   --shm-size=2g   --name medicalnet-train-cuerpoV-resnet50 \
  medical-net-devel bash -lc "source /setup/venv/bin/activate && \
  python train_classification.py data/cuerpoV/ sin_fractura.txt con_fractura.txt -m resnet50 -e 300"
podman wait medicalnet-train-cuerpoV-resnet50

podman run -d   -v /home/arauzo/MedicalNet/:/work-dir   --gpus=all   --shm-size=2g   --name medicalnet-train-cuerpoV-resnet50-f \
  medical-net-devel bash -lc "source /setup/venv/bin/activate && \
  python train_classification.py data/cuerpoV/ sin_fractura.txt con_fractura.txt -m resnet50 -f -e 300"
podman wait medicalnet-train-cuerpoV-resnet50-f
