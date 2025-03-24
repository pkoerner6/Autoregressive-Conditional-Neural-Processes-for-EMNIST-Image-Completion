# Extending Autoregressive Conditional Neural Processes to EMNIST Image Completion

This repository extends the evaluation of the paper:

**Autoregressive Conditional Neural Processes**  
_Wessel P. Bruinsma, Stratis Markou, James Requiema, Andrew Y. K. Foong, Tom R. Andersson, Anna Vaughan, Anthony Buonomo, J. Scott Hosking, Richard E. Turner_  
[arXiv:2007.08146]([https://arxiv.org/abs/2007.08146](https://arxiv.org/abs/2303.14468))

to the task of image completion on the **EMNIST** dataset of handwritten digits and characters.


## Overview

In the original paper, Autoregressive Neural Processes were evaluated primarily on regression-style tasks. This project expands the experimental scope to include:

- **EMNIST image completion**
- Comparative evaluation of **ConvCNPs**, **ConvGNPs**, and **AR-ConvCNPs**
- Visual and quantitative analysis of model performance with varying context set sizes
