# Reproducing MAML

## Abstract

This is an attempt to reproduce the results of the MAML paper of [Finn et al (2017)[1]](https://arxiv.org/abs/1703.03400), the need of which is explained in [2]. Maml is a meta-learning algorithm that learns to generalize from a few data points. It's about how a neural network in the K-shot regime learns and stays fit, ie. neither under nor overfit. The fundamental premise in this paper is that parameter initialization has important influence over model learning. A good initialization makes it possible for rapid learning and generalization. 

My key finding is that under a specific generator settings, their supervised regression result is reproducible. However, I found that maml deteriorates quickly as the training data deviates from the standard form. It is unable to extend its excellent performance to certain common periodic functions, a disappointing conclusion but I think it's not unexpected. 

## Introduction

MAML, short for Model-Agnostic Meta-Learning, finds patterns in learning that could be extended to problems that it's never seen before.

I'm only interested in supervised regression here as there are far more regression problems than classificaiton ones in the real world. And regression problems are generally harder.

## Reproduction Experiment

To reproduce a published result, I believe it's not enough to just run the github code on a given datasets. Here, I take the approach that requires an algorithm to do well on an out-of-sample basis. Similar to an ablation study, I identify the regimes where the model, in this case the meta-model is sensitive and how how far could I go before learning fails.



## Data


## Results


## Conclusion


## Code
The code is taken from this repo (https://github.com/cbfinn/maml). 

### Dependencies
This code has been tested on the following:
* python 3.7.3
* TensorFlow v1.13.1

### Data


### Usage
To run the code, see the usage instructions at the top of `main_fun.py`.

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/htso/maml_reproduction/issues).


References

[1] Finn, C., Abbeel, P., & Levine, S. (2017, August). Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (pp. 1126-1135). JMLR. org.

[2]"Is There a Reproducibility Crisis in Science?". Nature Video, Scientific American. 28 May 2016. (https://www.scientificamerican.com/video/is-there-a-reproducibility-crisis-in-science/)





