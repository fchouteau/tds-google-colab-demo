# Google Colab & Deep Learning VM

## Google Deep Learning VM

```bash
export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="europe-west1-d"
export INSTANCE_NAME="fch-tds"

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

## Slides

```bash
reveal-md tds.md -w --css static/reveal.css  --highlight-theme solarized-dark

reveal-md tds.md --css static/reveal.css --static=site --static-dirs=static  --highlight-theme solarized-dark

reveal-md tds.md --print slides.pdf -css static/reveal.css --highlight-theme solarized-dark
```
