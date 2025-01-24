# TF-IDF

Term Frequency - Inverse Document Frequency is a well known method for representing a document as a bag of words. For a given corpus $\mathbb{C}$, we compute the IDF value for each word $w$ by taking $idf_w := \frac{1}{log df_w}$, with $df_w$ denoting the number of documents in $\mathbb{C}$ containing the word $w$. The document $d$ is represented by a vector of length corresponding to the number of unique words in $\mathbb{C}$. Each element of the vector will be a tf-idf value for the word, i.e. $tfidf_w^d := tf_w^d \cdot idf_w$, where $tf_w^d$ represents the term frequency of the word $w$ in document $d$. Sometimes, we may l1 or l2 normalize the tf-idf vector so that the dot product between document vectors represents the cosine similarity between them.

## Bayesian Smoothing

We may want to apply some bayesian smoothing to the $tf_w^d$ terms to avoid spurious matches. For example, suppose that a rare word $w_r$ appears only in documents $d_1$ and $d_2$ in the entire corpus just by random chance. The $idf_{w_r}$ will be a large value, and hence documents $d_1$ and $d_2$ will have a high cosine similarity just because of this rare word.

For the specific setting I am considering, we can deal with this problem using bayesian smoothing. The setting is as follows: 
- Each document $d$ represents a job, and each job is tagged to an occupation $o$
- An occupation can have one or more jobs tagged to it
- We wish to represent each occupation as a TF-IDF vector of words

To apply bayesian smoothing to this scenario, notice that we only need to smooth the term frequencies $tf_w^d$. Since the IDF values $idf_w$ are estimated across the whole corpus, we can assume that those are relatively reliable. And since term frequencies are counts, we can use a poisson random variable to represent them. See [reference](https://www.y1zhou.com/series/bayesian-stat/bayesian-stat-bayesian-inference-poisson/) for a primer on the gamma-poisson bayesian inference model.

Specifically, we assume that $\theta_w^o$ is the poisson parameter that dictates the term frequency of $w$ in any document belonging to $o$, i.e. $tf_w^d \sim Poisson(\theta_w^o)$. We treat the observed term frequency $tf_w^d$ for each document $d$ belonging to $o$ as a data sample to update our beliefs about $\theta_w^o$. We start with an uninformative gamma prior for $\theta_w^o$, and obtain the MAP estimate for $\hat{\theta}_w^o$ as below, with $df_{d \in o}$ denoting the number of documents that belong to occupation $o$.

$$
    \hat{\theta}_w^o = \frac{a + \sum_{d \in o} tf_w^d - 1}{b + df_{d \in o}}
$$

We can thus use this formula to obtain posterior estimates for each $\hat{\theta}_w^o$. One possible choice of the prior parameters $a$ and $b$ is to set $a$ to be the mean term frequency for word $w$ per document in the entire corpus, and to set $b := 1$. This prior corresponds to $\hat{\theta}_w^o$ following a gamma distribution with mean $a$ and variance $a$, which seems to be a reasonable choice that can be overrided by a reasonable amount of data.

The posterior variance, which may also be helpful in quantifying the confidence of this estimate, is:
$$
    Var(\hat{\theta}_w^o) = \frac{a + \sum_{d \in o} tf_w^d}{(b + df_{d \in o})^2}
$$

Finally, after obtaining the posterior estimates for each $\hat{\theta}_w^o$, we can just use them as our term frequencies and multiply them by the IDF values as per normal. We can also apply l1 or l2 normalization thereafter to the tf-idf vectors. This method should produce tf-idf vectors that are more robust to the original problem of spurious matches. 

For illustration, for a very rare word $w_r$, $a$ will be a low value close to 0 (say 0.01). Suppose we were to observe $n$ number of new documents, each containing one occurrence of word $w_r$. Then the posterior estimate of $\hat{\theta}_w$ will update as follows:

| n   | $\hat{\theta}_w$ |
| --- | ---------------- |
| 1   | 0.005 |
| 2   | 0.337 |
| 3   | 0.503 |
| 4   | 0.602 |
| $...$ | $...$   |
| 20  | 0.905 |

As desired, the estimate for $\hat{\theta}_w$ starts off at a very small value and gradually approaches the true value $1$. This will help mitigate the effect of spurious matches. If we desire for the update to match the data more quickly, we can simply scale $a$ and $b$ down by some factor, e.g. now $a := \frac{a}{5} = 0.002$ and $b := \frac{b}{5} = 0.2$. Then we have:

| n   | $\hat{\theta}_w$ |
| --- | ---------------- |
| 1   | 0.001 |
| 2   | 0.455 |
| 3   | 0.626 |
| 4   | 0.715 |
| $...$ | $...$   |
| 20  | 0.941 |

As a final detail, note that the update formula can result in negative estimates if $a < 1$ and $\sum_d tf_w^d = 0$. The small negative value is probably not a big problem for our purposes, but we could also resolve it by setting the negative values to zero if desired.