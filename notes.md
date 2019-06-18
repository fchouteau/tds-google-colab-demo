# Google Colab & Deep Learning VM

## Eurosat Dataset

http://madm.dfki.de/downloads
https://drive.google.com/open?id=1dftmi50__aE2GhW6eB0C8DToFnJD7l2e

## Google Deep Learning VM

```bash
export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="europe-west1-d"
export INSTANCE_NAME="tds-demo"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-p100,count=1" \
  --metadata="install-nvidia-driver=True" \
  --boot-disk-size=120GB \
  --preemptible
```
https://github.com/gclouduniverse/gcp-notebook-executor
https://github.com/gclouduniverse/nova-jupyterlab-extensions

### How to launch from the CLI & get jupyterlab

https://towardsdatascience.com/how-to-use-jupyter-on-a-google-cloud-vm-5ba1b473f4c2

### Google Cloud Notebook Executor

```bash
# You can use any branch but this article been tested with 0.1.2 only
git clone https://github.com/gclouduniverse/gcp-notebook-executor.git --branch v0.1.3

cd gcp-notebook-executor
source utils.sh

INPUT_NOTEBOOK="gs://{your-storage}/ai-notebook-demo.ipynb"
GCP_BUCKET="gs://{your-storage}/runs"
IMAGE_FAMILY_NAME="pytorch-latest-gpu"
INSTANCE_TYPE="n1-standard-8"
GPU_TYPE="k80"
GPU_COUNT=1
ZONE="europe-west1-b"

execute_notebook -i "${INPUT_NOTEBOOK}" \
                 -o "${GCP_BUCKET}" \
                 -f "${IMAGE_FAMILY_NAME}" \
                 -t "${INSTANCE_TYPE}" \
                 -z "${ZONE}" \
                 -g "${GPU_TYPE}" \
                 -c "${GPU_COUNT}"
```


## Slides

```bash
reveal-md README.md -w --css static/style.css

cp README.md tds.md && reveal-md tds.md --css static/style.css --static=docs --static-dirs=static --absolute-url https://fchouteau.github.io/tds-google-colab-demo

cp README.md tds.md && reveal-md tds.md --print slides.pdf --css static/reveal.css
```
