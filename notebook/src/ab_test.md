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


## References

- [Experiments at Airbnb](https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7)
- [Evan Miller's AB Testing Tools](https://www.evanmiller.org/ab-testing/)
- [PostHog AB Testing Examples](https://posthog.com/product-engineers/ab-testing-examples)
- [How we experiment at Monzo](https://monzo.com/blog/2022/05/24/pellets-not-cannonballs-how-we-experiment-at-monzo)
- [Ron Kohavi 2022 - AB Testing Intuition Busters](https://drive.google.com/file/d/1oK2HpKKXeQLX6gQeQpfEaCGZtNr2kR76/view)
- [Convoy - The Power of Bayesian AB Testing](https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5)