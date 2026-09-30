# Implementing Self-Distillation with No Labels for Lymphoma Biopsies in Julia

> [!NOTE]
> This repository is currently under construction and may be incomplete. The remaining code, scripts and documentation will be uploaded shortly.

## Abstract
Automated cancer sub-typing from lymphoma biopsy whole slide images (WSIs) could accelerate the diagnostic process and enable more educated treatment decisions.
WSIs are commonly divided into individual tiles to make them compatible with standard computer vision pipelines.
Those local tiles, however, do not necessarily correspond well to slide-level labels.
This makes training traditional supervised classifiers challenging.
Self-supervised approaches have shown success in learning superior image features without relying on labels.
DINOv2, a method for training self-supervised vision foundation models, trains a student network alongside a mean teacher network to arrive at meaningful image representations.
This approach has been successfully applied to a broad dataset of WSI tiles from diverse origins.
Lymphoma biopsy WSIs might benefit from a model trained on more specific data.
In this thesis, we implement DINOv3, a revision of DINOv2, in Lux.jl, a pure Julia deep-learning
framework.
We adapt the existing method to run on fewer devices with smaller batch sizes.
We propose continued pre-training from available DINOv3 checkpoints trained on general
web images (LVD-1689M).
Initializing the student and teacher backbones from existing parameters allows us to leverage the training already performed for DINOv3.
We were unable to evaluate our approach as training aborted due to a memory leak that accumulates over iterations.
Analysis shows that the presented implementation suffers from increased compilation overhead. Profiling suggests that later iterations run at a constant pace.

Please refer to the whole [thesis](documents/thesis.pdf) for more information.

## Getting Started
Please follow the [official instuctions](https://julialang.org/downloads/) to download and install `juliaup` and `julia`. Once installed open up the REPL and instantiate the environment
```julia-repl
julia> # Press ] to enter pkg mode

(@v1.13) pkg> activate .

(BachelorThesis) pkg> instantiate
```