# Gadolinium-Free Cardiac MRI Myocardial Scar Detection by 4D Convolution Factorization

## Description of Code

This python code implements our Spatiotemporal residual attention network (ST-RAN) for the detection of scar from non-enhanced sax-cine images.
The models were implemented using Tensorflow 2.4.1.

In architectures, 2 sets of models are available:
- Ablation: codes for the ablation study. The ablation study includes Spatial network, spatiotemporal network, spatiotemporal network with dual attention, and with residual attention.
- SOTA: codes for comparison with state-of-the-art methods. The architectures include 3D convolutional neural network (CNN), CNN with long short term memory, and 2D spatiotemporal network.

In main, there is also 2 folders:
- Ablation: training using ablation architectures. One main file per architecture.
- SOTA: training using SOTA architectures. One main file per architecture.


## Abstract

Gadolinium-based contrast agents are commonly used in cardiac magnetic resonance (CMR) imaging to characterize myocardial scar tissue. Recent works using deep learning have shown the promise of contrast-free short-axis cine images to detect scars based on wall motion abnormalities (WMA) in ischemic patients. However, WMA can occur in patients without a scar. Moreover, the presence of a scar may not always be accompanied by WMA, particularly in non-ischemic heart disease, posing a significant challenge in detecting scars in such cases. To overcome this limitation, we propose a novel deep spatiotemporal residual attention network (ST-RAN) that leverages temporal and spatial information at different scales to detect scars in both ischemic and non-ischemic heart diseases. Our model comprises three primary components. First, we develop a novel factorized 4D (3D+time) convolutional layer that extracts 3D spatial features of the heart and a deep 1D kernel in the temporal direction to extract heart motion. Secondly, we enhance the power of the 4D (3D+time) layer with spatiotemporal attention to extract rich whole-heart features while tracking the long-range temporal relationship between the frames. Lastly, we introduce a residual attention block that extracts spatial and temporal features at different scales to obtain global and local motion features and to detect subtle changes in contrast related to scar. We train and validate our model on a large dataset of 3000 patients who underwent clinical CMR with various indications and different field strengths (1.5T, 3T) from multiple vendors (GE, Siemens) to demonstrate the generalizability and robustness of our model. We show that our model works on both ischemic and non-ischemic heart diseases outperforming state-of-the-art methods. 