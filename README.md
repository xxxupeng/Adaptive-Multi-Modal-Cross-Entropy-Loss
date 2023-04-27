# Rethinking Ground-Truth Distributions in Cross-Entropy Loss for Stereo Matching

## Abstract

Despite the great success of deep learning in stereo matching, recovering accurate and clearly-contoured disparity map is still challenging. Currently, L1 loss and Cross-entropy loss are the two most widely used loss functions for training the stereo matching networks. Comparing with the former, the Cross-entropy loss can usually achieve better results thanks to its direct constrain to the the cost volume. However, how to generate reasonable ground-truth distribution for this loss function remains largely under exploited. Existing works assume uni-modal distributions around the ground-truth for all of the pixels, which ignores the fact that the edge pixels may have multi-modal distribution. In this paper, we first experimentally exhibit the importance of correct edge supervision to the overall disparity accuracy. Then a novel adaptive multi-modal cross-entropy loss which encourages the network to generate different distribution patterns for edge and non-edge pixels is proposed. We further optimize the disparity estimator in the inference stage to alleviate the bleeding and misalignment artifacts at the edge. Our method is generic and can help classic stereo matching models regain competitive performance. GANet trained by our loss ranks 1st on the KITTI 2015 and 2012 benchmarks and outperforms state-of-the-art methods by a large margin. Meanwhile, our method also exhibits superior cross-domain generalization ability and outperforms existing generalization-specialized methods on four popular real-world datasets.

## Environment

- python == 3.9.12
- pytorch == 1.11.0
- torchvision == 0.12.0
- numpy == 1.21.5

## Datasets

- [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [KITTI 2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)

Download the datasets, and change the `datapath` args. in `./scripts/sceneflow.sh` or `./scripts/kitti.sh`.

## Training

We use the distributed data parallel (DDP) to train the model.

Please execute the bash shell in `./scripts/`, as:

```bash
/bin/bash ./scripts/sceneflow.sh
/bin/bash ./scripts/kitti.sh
```

Training logs are saved in `./log/`.

Change `loss_func` args. for different losses:
- SL1: smooth L1 loss
- ADL: our Adaptive multi-modal Ground-truth distribution cross-entropy Loss

## Evaluation

Please uncomment and execute `val.py`.

`EPE`, `1px`, `3px`, `D1`, `4px`, `speed` are reported.

Change `postprocess` args. for different disparity distribution post-processing methods:
- mean: soft-argmax
- argmax: argmax
- SM: Single-Modal weighted average
- DM: our Dominant-Modal weighted average