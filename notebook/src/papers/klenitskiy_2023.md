# Klenitskiy 2023 - BERT4Rec vs SASRec

[Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?](https://arxiv.org/abs/2309.07602)

This is a short paper that argues the alleged performance gains of [BERT4Rec](https://arxiv.org/abs/1904.06690) over [SASRec](./kang_2018.md) is not due to the masked cloze prediction task and bi-direction attention as the authors claim, but rather due to the loss function. 

BERT4Rec uses softmax cross entropy over the entire item catalog at each time step, whereas SASRec uses binary cross entropy against a single sampled negative at each time step. When the same softmax cross entropy is used for SASRec, it outperforms BERT4Rec consistently and trains faster.

## Background

Sequential recommendation is a popular approach currently to recommender systems, and in particular transformer models with self attention are the standard approach. [SASRec](./kang_2018.md) is the standard approach, where the task of sequential recommendation is treated as a causal modelling task where the self attention mechanism is only allowed to attend to previous time steps when making the prediction at time step $t$. 

BERT4Rec was proposed as an improvement to SASRec, and the claim was that introducing bi-directional attention (like BERT) and performing prediction on the cloze passage task (i.e. randomly masking x% of items) is able to lead to significant gains over SASRec. The argument is that the random masking is akin to data augmentation, as there are far more permutations of masked positions compared to just predicting the next item.

The authors point out two misgivings they have with this interpretation, which is in line with my own intuitions:
- BERT4Rec task is only <<weakly related>> to the final goal of sequential recommendations, whereas SASRec tasks for training and prediction are perfectly aligned (i.e. just predict the next item). This is akin to only using the BERT encoder (without the decoder) for a language modelling task, which is quite strange.
- BERT4Rec masks some items and <<only calculates losses for the subset of items>>, whereas SASRec computes losses for all items (except the last item) at once, getting more training signal from each training sequence

So we should expect SASRec to perform better and more efficiently than BERT4Rec. How then do we explain the performance discrepancy? The authors hypothesize that the performance difference is really due to the difference in loss functions between them, as explained below.

## Setup 

Start with a set of users $\mathcal{U}$ and items $\mathcal{I}$. Let each user $u \in \mathcal{U}$ be represented by a sequence of item interactions $s_u = \{ i_1^{(u)}, i_2^{(u)}, ..., i_{n_u}^{(u)} \}$. Each sequential deep learning model may be abstracted as an encoder of input sequence $s_u$, and the encoded sequence be denoted $H_u \in \R^{n_u \times d}$, where $d$ is the latent dimension. 

To make predictions, given the full item embedding matrix $E \in \R^{|\mathcal{I}| \times d}$, we take:
$$
    R_u = H_uE^T \in \R^{n_u \times |\mathcal{I}|}
$$

Then the $t, i$ element of $R_u$ may be denoted as $r^{(u)}_{t,i}$ represents the predicted relevance of item $i$ at time step $t$ for user $u$.

<<SASRec: Binary cross entropy loss>>. SASRec does not compute the full $R_u$ prediction matrix. Instead, for each true positive item at each time step, it randomly samples one negative item and computes the predictions $r^{(u)}_{t,i_t}$ and $r^{(u)}_{t,-}$. Then the loss is:
$$
    \L_{BCE} = -\sum_{u \in \mathcal{U}} \sum_{t=1}^{n_u} \log(\sigma(r^{(u)}_{t,i_t})) + \log(1-\sigma(r^{(u)}_{t,-}))
$$

<<BERT4Rec: Softmax Cross Entropy>>. In contrast, BERT4Rec computes the full prediction matrix $R_u$ for each user and computes the softmax over the entire item catalog for each masked item prediction. The cross entropy loss is thus:
$$
    \L_{CE} = -\sum_{u \in \mathcal{U}} \sum_{t \in T_u} \log 
        \frac{
            \exp \left( r^{(u)}_{t,i_t} \right)
        }
        {
            \sum_{i \in \mathcal{I}} \exp \left( r^{(u)}_{t,i} \right)
        }
$$

Note that for BERT4Rec, the inner summation is only over the time steps with masked items. If we were to translate this loss to SASRec, we would sum over all time steps.

<<Sampled Softmax>>. Finally, it may not be computationally feasible to compute the softmax over the full item catalog. Hence the authors propose that for each user sequence in a batch, we sample $N$ items a user has not interacted with and use the same set of negatives for each time step of a given sequence. Let $\mathcal{I}_N^{u-}$ denote all items that user $u$ has not interacted with. The loss is then: 
$$
    \L_{CE-sampled} = -\sum_{u \in \mathcal{U}} \sum_{t = 1}^{n_u} \log 
        \frac{
            \exp \left( r^{(u)}_{t,i_t} \right)
        }
        {
            \exp \left( r^{(u)}_{t,i_t} \right) +
            \sum_{i \in \mathcal{I}_N^{u-}} \exp \left( r^{(u)}_{t,i} \right)
        }
$$

> Qn: This means that we have a different set of negatives for each user in a batch? Seems quite memory intensive.

## Experiments

The authors use the full sequence of item interactions for each user. The last (most recent) item is held out as the test set, and the second last item is chosen as the validation step. Models are trained with early stopping on the validation set. The authors note that the common practice of sampling negative items for metric computation is not a robust one, as it introduces randomness into the metrics.

> Note: I think the other reason sampling negatives is not robust is because it does not directly mirror the retrieval task, which requires choosing an item from the full catalogue.

The results show that SASRec with `3,000` negatives is consistently the best model, beating BERT4Rec consistently. It also trains around $\frac{1}{2}$ to $\frac{1}{4}$ times faster than BERT4Rec. Hence the authors recommend sampled softmax SASRec as the de-facto standard instead of BERT4Rec.

The authors do note that for the smaller datasets, SASRec may overfit relatively quickly (validation loss peaks and declines) and hence it is important to use early stopping. In contrast, BERT4Rec is more robust to overfitting and validation performance generally does not decline (it plateaus near the peak).