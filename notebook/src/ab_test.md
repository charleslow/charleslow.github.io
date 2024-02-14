# AB Testing

These summaries are based on reading [PostHog's article on AB testing](https://posthog.com/product-engineers/ab-testing-examples) and studying the ones of interest further.

## AB Testing at Airbnb

AB Testing is crucial because the outside world often has a larger effect on metrics than product changes. Factors like seasonality, economy can cause metrics to fluctuate greatly, hence a controlled experiment is necessary to control for external factors and isolate the effects of the product change.

Airbnb has a complex booking flow of `search` -> `contact` -> `accept` -> `book`. While they track the AB impact of each stage, the main metric of interest is the search to book metric.

One pitfall is <stopping the AB Test too early>. Airbnb noticed that their AB tests tend to follow a pattern of hitting significance early on but returning to neutral when the test has run its full course. This is a phenomenon known as <peeking>, which is to repeatedly examine an ongoing AB test. It makes it much more likely to find a significant effect when there isn't, since we are doing a statistical test each time we peek. For Airbnb, they hypothesize that this is also a phenomenon caused by the long lead time it takes from `search` -> `book`, such that early converters have a disproportionately large influence at the beginning of the experiment.

The natural solution to this problem is to conduct power analysis and determine the desired sample size prior to the experiment. However, Airbnb runs multiple AB tests at the same time, hence they required an automatic way to track ongoing AB tests and report when significance has been reached. Their solution back then was to create a `dynamic p-value graph`. The idea is that on day 1 of the experiment, we would require a very low p-value to declare success. As time goes on and more samples are collected, we can gradually increase the p-value until it hits 5%. The shape of this graph is unique to their platform and they did extensive simulations to create it, so this solution is not very generalizable.

Another pitfall was <assuming that the system is working>. After running an AB test for shifting from more words to more pictures, the initial result was neutral. However, they investigated and found that most browsers had a significant positive effect except for Internet Explorer. It turned out that the change had some breaking effect on older IE browsers. After fixing that, the overall result became positive. Hence some investigation is warranted when the AB test results are unintuitive. However, one needs to be cautious of multiple-testing when investigating breakdowns, since we are conducting multiple statistical tests.

Airbnb has a strong AB testing culture - only 8% of AB tests are successful (see Ron Kohavi's [LinkedIn Post](https://www.linkedin.com/posts/ronnyk_interesting-discussion-in-thread-below-about-activity-7114309358942900224-X26v)).

## AB Testing at Monzo

Monzo has a big AB testing culture - they ran 21 experiments in 6 months. Monzo has a bottom-up AB testing culture where anyone can write a proposal on Notion. Some of the best ideas come from the customer operations staff working on the frontlines. A proposal comprises the following sections:
- What problem are you trying to solve?
- Why should we solve it?
- How should we solve it (optional)?
- What is the ideal way to solve this problem (optional)?

Many proposals end up becoming AB experiments. Monzo prefers launching <pellets> rather than <cannonballs>. This means that each experiment comprises small changes, is quick to build, and helps the team learn quickly.

## AB Testing at Convoy

Convoy argues that bayesian AB testing is more efficient than frequentist AB testing and allows them to push out product changes faster while still controlling risk. 

The argument against frequentist AB testing is as follows. Under traditional AB testing, we define a null hypothesis using the control group (call it A), and declare a treatment (call it B) as successful if the treatment value has a significant p-value, i.e. it falls outside of the range of reasonable values under the null. Based on power analysis and an expected effect size, we predetermine the necessary sample size to achieve sufficient power, and once this sample size is reached, we have a binary `success` or `failure` result based on the p-value.

Convoy argues that this approach is safe but inefficient. This is because prior to the sample size being reached, we do not have a principled way of saying anything about the effectiveness of the treatment, even if it is performing better. Furthermore, frequentist AB testing gives us a binary result, but it does not quantify the size of the difference. Specifically, an insignificant test where `E(A)=10%, E(B)=11%` is quite different from `E(A)=15%, E(B)=10%`. For the former case, one can argue for launching B even if the p-value did not hit significance, whereas for the latter we should definitely not launch.

Bayesian analysis comes in to make the above intuition concrete. Suppose we are interested in the clickthrough rate (CTR) of variant A vs B. Bayesian analysis provides a distribution of the average CTR for each variant `A, B` at any point of the AB test, based on the results that it has seen thus far. These posterior distributions reflect both the mean of the data (how far apart $E(A)$ is from $E(B)$) and the variance of the data (how spread out the distributions are), allowing us to quantify how much we stand to gain if we were to pick either variant A or B at this point in time.

Concretely, they define a loss function as follows. Let $\alpha$ and $\beta$ be the unobserved true CTR for variants `A` and `B` respectively, and let the variable $x$ denote which variant we decide to choose. Then our loss for choosing each variant can be expressed as:

$$
\begin{align*}
    \mathcal{L}(\alpha, \beta, x) = 
\left\{ 
    \begin{array}{ c l }
        max(\beta - \alpha,\ 0) \quad \textrm{if } & x = A\\
        max(\alpha - \beta,\ 0) \quad \textrm{if } & x = B\\
    \end{array}
\right.
\end{align*}
$$

In other words, the loss above expresses how much we <stand to lose> by picking the unfortunately wrong variant based on incomplete evidence at this point in time. Of course, we do not know the true values of $\alpha$ and $\beta$, so we need to estimate the loss using our posterior distributions which we computed from data. We then compute the *expected* loss based on the posterior distributions $\hat{\alpha} \sim A$, $\hat{\beta} \sim B$ as such:
$$
    \mathbf{E}(\mathcal{\hat{L}}) = \int_{\hat{\alpha} \sim A} \int_{\hat{\beta} \sim B} \mathcal{L}(\hat{\alpha}, \hat{\beta}, x) \cdot f(\hat{\alpha}, \hat{\beta}) \quad d\alpha \ d\beta
$$

Here, $f(\hat{\alpha}, \hat{\beta})$ is the joint posterior distribution, which I believe we can obtain by multiplying the two independent posterior distributions $\hat{\alpha}$, $\hat{\beta}$ together. We can also perform random draws from the posterior distributions to estimate this statistic. Finally, we make a decision by choosing the variant that dips below a certain loss threshold, which is usually a very small value.

The appeal of the bayesian approach is two-fold:
1. It allows us to make <faster decisions>. Suppose an experiment is wildly successful, and it is clear within a day that variant B is better. Bayesian analysis will be able to reveal this result, whereas frequentist analysis will tell us to wait longer (since we estimated the effect size to be smaller).
2. It allows us to <control risk>. Since we are making decisions based on minimizing risk (supposing we had picked the poorer variant), we may be sure that even if we are wrong, it will not severely degrade our product. So supposing that there is no significant engineering cost between variant A and B, we can more rapidly roll out new iterations with the assurance that on average our product will be improving.

## Power Analysis

Reference: [Probing into Minimum Sample Size by Mintao Wei](https://towardsdatascience.com/probing-into-minimum-sample-size-formula-derivation-and-usage-8db9a556280b)

How to determine the minimum sample size required to achieve a certain significance level and power desired?

The following table helps us understand how type I and type II errors come into play:

| | Null Hypothesis: A is True | Alternate Hypothesis: B is True |
|:------:|:-------:|:--------:|
|Reject A| Type I Error | Good statistical power |
|Accept A| Good significance level | Type II Error |

<Type I Error> refers to rejecting the null hypothesis when it is actually true, e.g. when we think that an AA test has significant difference. In short, it means we were <too eager to deploy a poor variant>. This should happen with probability $\alpha$, which is the significance level which we set (typically 5%). We have a better handle on type I error because the baseline conversion rate is typically known prior to an experiment. 

<Type II Error> refers to failing to reject the null hypothesis when the alternate is actually true, i.e. we failed to get a significant effect on an improvement that is *known* to be better. In short, we were <too conservative and failed to deploy a winning variant>. In order to reason about type II error, we need to make a guess on what is the distribution of test variant `B`. Typically, this is done by assuming a minimum effect $\delta$ we wish to detect, and setting $\mu_B = \mu_A + \delta$, and re-using the standard deviation from `A`. With these assumptions in place, we use $power = 1-\beta$ to determine the type II error that should only occur with probability $\beta$ (typically 20%). Note that since $\delta$ is the *minimum* effect we wish to detect, if the actual effect turned out to be larger, the type II error can only be smaller than our desired amount, which is ok.

Now we can derive the formula for the minimum sample size required to achieve the desired levels of type I and type II error respectively.

Let us define the baseline conversion rate as $p$, and the minimum relative detectable effect rate as $d$. Consequently, the minimum detectable delta is $\delta = d \times p$. Let the desired power level be $1-\beta$, and the desired significance level as $\alpha$. Assume the scenario where we are running an AA or AB test with two variants of sample size $N$ each.

Firstly, we write down the distribution of the sample mean difference supposing we knew the true population means and standard deviations. Let $\mathbb{E}(X_A) = \mu_A, Var(X_A) = \sigma_A^2$ and $\mathbb{E}(X_B) = \mu_B, Var(X_B) = \sigma_B^2$. Note that $X_A, X_B$ may have arbitrary distributions, e.g. they could measure proportions, revenue etc.

Under the central limit theorem, the sample means will be distributed like so with $N$ samples: $\bar{X}_A \sim N \left( \mu_A, \frac{\sigma_A^2}{N} \right)$, $\bar{X}_B \sim N \left( \mu_B, \frac{\sigma_B^2}{N} \right)$. Importantly, the difference of the sample means will have the distribution below. Note that we add the variances together because $Var(B-A) = Var(B) + Var(A)$ for any two independent random variables $A, B$.

$$
    \bar{X}_{D} = \bar{X}_B - \bar{X}_A \sim N \left( \mu_B - \mu_A, \frac{\sigma_A^2 + \sigma_B^2}{N} \right)
$$

Now we can start working from the desired $\alpha, \beta$ levels to the minimum sample size. We need to ensure that both objectives below are achieved with our sample size $N$:
1. Assuming null hypothesis to be true, ensure that type I error $\leq \alpha$.
2. Assuming alternate hypothesis to be true, ensure that type II error $\leq 1 - \beta$.

Our overall strategy is thus:
1. Write the true distribution of sample mean difference $\bar{X}_D$ as a function of $N$
2. Write an equation that enforces objective 1
    - Assume the null hypothesis is true
    - This equation should express that the probability of the observed sample mean difference $\bar{x}_D$ falling in the critical region of the distribution $\bar{X}_D$ is $\leq \alpha$.
    - This should simply enforce that we set the critical value at $z_{\alpha | \bar{X}_D, H_0}$ 
    - We cannot set it any lower because we will violate objective 1
3. Write an equation that enforces objective 2
    - Now importantly, the minimal detectable effect $\delta$ must lie beyond the critical value derived earlier.$z_{\alpha | \bar{X}_D, H_0}$ Why? Suppose otherwise. Then 


Let us first tackle objective 1. Assuming the null hypothesis is true, observe that $\bar{X}_D \sim N \left( 0, \frac{\sigma_A^2 + \sigma_B^2}{N} \right)$. For a given level of significance $\alpha$, under the null hypothesis, denote the critical value for rejecting the null as $z_{\alpha|\bar{X}_D, H_0}$. Now, under the null hypothesis, the probability of committing a type I error is:

$$
\begin{align*}
    P \left( \bar{X}_{D} \geq z_{H_0}(\alpha) \right) &= 
    P \left( Z \geq \frac{\bar{x}_{D} - 0}{\sqrt{\frac{\sigma_A^2 + \sigma_B^2}{N}}} \right)
\end{align*}
$$

Notice that the minimal effect $\delta$ should not enter into this equation, because type I error simply assumes the null hypothesis, so the alternate hypothesis does not and should not show up here.

Since we wish to control the type I error to be $\leq \alpha$ by controlling $N$, we derive the following:

$$
\begin{align*}
    P \left( Z \geq \frac{\bar{x}_{D}}{\sqrt{\frac{\sigma_A^2 + \sigma_B^2}{N}}} \right) &\leq \alpha\\
    z_{H_0}(\alpha) \leq \frac{\delta}{\sqrt{\frac{\sigma_A^2 + \sigma_B^2}{N}}}\\
    N \geq \frac{z_{H_0}^2 \left[ \sigma_A^2 + \sigma_B^2 \right] }{\delta^2}
\end{align*}
$$

Line 2 of the above is saying that we wish the critical value of the distribution to be *less* than the minimal detectable effect

We write down the probability of wrongly rejecting the null hypothesis (i.e. committing a type I error):

$$

$$

## References

- General:
    - [Experiments at Airbnb](https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7)
    - [Evan Miller's AB Testing Tools](https://www.evanmiller.org/ab-testing/)
    - [PostHog AB Testing Examples](https://posthog.com/product-engineers/ab-testing-examples)
    - [How we experiment at Monzo](https://monzo.com/blog/2022/05/24/pellets-not-cannonballs-how-we-experiment-at-monzo)
    - [Ron Kohavi 2022 - AB Testing Intuition Busters](https://drive.google.com/file/d/1oK2HpKKXeQLX6gQeQpfEaCGZtNr2kR76/view)
    - [Francesco Casalegno - AB Testing A Complete Guide](https://towardsdatascience.com/a-b-testing-a-complete-guide-to-statistical-testing-e3f1db140499)

- On Bayesian AB Testing:
    - [Convoy - The Power of Bayesian AB Testing](https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5)
    - [Chris Stucchio's White Paper](https://www.chrisstucchio.com/pubs/VWO_SmartStats_technical_whitepaper.pdf)
    - [VeepeeTech Intro to Bayesian AB Testing](https://medium.com/vptech/introduction-to-bayesian-a-b-testing-in-python-df81a9b3f5fd)