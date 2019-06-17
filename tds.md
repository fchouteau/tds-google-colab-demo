---
title: Google Colab & AI Notebook
theme: solarized
highlightTheme: solarized-dark
separator: ---
verticalSeparator: --
revealOptions:
    transition: 'fade'
    transitionSpeed: 'default'
    controls: false
---

# Google Colaboratory
#  & AI Notebook

## Data Science in the cloud, the easy way

Toulouse Data Science #38 - June 18th 2019

Florient CHOUTEAU

--

### about.md

<img src="static/red-panda.png" alt="" width="168px" height="146px" style="background:none; border:none; box-shadow:none; position: fixed; top: 10%; right: 20%;"/>

- ML Engineer @ Airbus Defence and Space (we are hiring !)

- Training Neural Networks since 2016 

    - Remote sensing imagery

    - Delair, Magellium / Airbus Intelligence (**spoilers**), Airbus DS...

    - torch, tf, keras, pytorch, ...

- Contact: [@foxchouteau](https://twitter.com/foxchouteau) or on Slack

--

### Who started learning data science recently ?

--

### Who works in data science ?

--

### Who teaches data science classes ?

--

### TL;DR

- easy access to configured development environment for ML

- from Google but not limited to their tech

- jupyter-based products

- one free, one paid: different use cases, similar principles

--

### Disclaimer

This talk is not sponsored by Google ;)

I didn't check every alternative product: Some may be better

---

## Colaboratory

<img src="static/colab.png" alt="" style="background:none; border:none; box-shadow:none;"/>

--

<img src="static/open_in_colab.png" alt="" style="background:none; border:none; box-shadow:none;"/>

---

### WTF is... Google Colab ?

- Jupyter Notebook + Google Drive

- Full python data science environment

- Somewhat long duration (12h at most)

--

### Is it for YOU ? 

- Students, self-learning

- Quick experiments / colaboration

--

### Interesting Features 

- Access to google drive data

- Can upload / download directly from colab 

--

### Interesting Features

- GPU ! (Nvidia T4 = 3000$)

- Collaboration ! (sharing notebooks)

- From github to colab
    - In github: https://github.com/{repo}/
    - In colab: https://colab.research.google.com/{user}/{repo}/{url_to_notebook}

--

### Demo !

- End-to-end training w/ GPU. Pytorch and ignite (**spoilers**)

- Data on Google Drive

- Saving model locally

https://colab.research.google.com

--

### Limitations

- Long calculations w/ guarantees

- Full control over installation and data

---

## Deep Learning VM / AI Platform Notebook

<img src="static/aiplatform.png" alt="" width="200px" height="200px" style="background:none; border:none; box-shadow:none;"/>

--

### WTF is... AI Platform Notebook ?

- Pre configured Cloud Virtual Machines (Google Compute Engine)

- With jupyter lab auto launched & ready

- Papermill pre installed for scheduling

--

### Available configurations

<img src="https://cdn-images-1.medium.com/max/800/1*TiNNFQ5encexSKn8hitRug.png" alt="" style="background:none; border:none; box-shadow:none;"/>

--

### 2 different workflows

A) Jupyter only ("AI Notebook")

B) Pre-configured instance for Data Science ("Deep Learning VM")

--

### Demo 1: "AI Platform Notebook"

- Creating an instance

- Connecting to jupyter lab (with or without ssh !)

https://console.cloud.google.com

--

### Demo 2: "Deep Learning VM"

- Using the DL VM as a preconfigured headless code runner

- Executing a notebook on a deep-learning-vm

```
INPUT_NOTEBOOK="gs://fchouteau-storage/ai-notebook-demo.ipynb"
GCP_BUCKET="gs://fchouteau-storage/runs"
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

--

### Advanced Usage (not covered here)

https://towardsdatascience.com/how-to-use-jupyter-on-a-google-cloud-vm-5ba1b473f4c2

- Preemptible (> 5x less expensive, run only 24h)
- CLI creation for more customization
- Scheduled execution w/ papermill

---

## Conclusion

-- 

### TL;DR (bis)

 <table style="width:100%">
  <tr>
    <th>Google Colab</th>
    <th>Google AI Notebook</th>
  </tr>
  <tr>
    <td>Learn, experiment </td>
    <td>Can scale compute</td>
  </tr>
  <tr>
    <td>Single notebook / Clone from github</td>
    <td>Upload own code</td>
  </tr>
  <tr>
    <td>Simple jupyter env.</td>
    <td>Full jupyter lab or SSH access</td>
  </tr>
  <tr>
    <td>Data from anywhere / google drive</td>
    <td>Fully owned cloud environment</td>
  </tr>
  <tr>
    <td>Short runtimes</td>
    <td>Cheap 1d runtimes or arbitrary runtimes</td>
  </tr>
  <tr>
    <td>**free**</td>
    <td>**paid** (by minute of computing + storage)</td>
  </tr>
</table> 

--

### Alternatives

- Kaggle Kernels (free ! Alternative to colab)

- Amazon Sagemaker

- A lot of companies with to-rent servers

- Build your own machine ? (opinion: last step for individual use)

--

### Thank you ! 