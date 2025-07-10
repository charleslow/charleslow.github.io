# Conformal Predictions

## Exchangeability

In this section we discuss the difference between independence and exchangeability.

> **Definition**. Independence.
> 
> Two events $A$ and $B$ from a sample space $S$ are said to be independent if $P(A \text{ and } B) = P(A)P(B)$.

If $P(B) > 0$, it is equivalent to say $P(A|B) = P(A)$, since:
$$P(A|B) = \frac{P(A, B)}{P(B)} = \frac{P(A)P(B)}{P(B)} = P(A)$$

> **Definition.** Exchangeability
> 
> Two events $A$ and $B$ are said to be exchangeable if $P(A_1=a, A_2=b) = P(A_1=b, A_2=a)$, which means there is indifference with respect to the order of events.
> 
> More generally, exchangeability of a sequence of events $A_1, ..., A_n$ means that the joint distribution is unchanged when we permute the order of events:
$$
    P(A_1, ..., A_n) = P(A_{\sigma(1)}, ..., A_{\sigma(n)}), \text{ for all permutations } \sigma
$$

The simplest way to understand is to use the example of drawing balls from an urn *without replacement* (example taken from Cordani 2006). Suppose we have an urn with `10 red balls` and `5 white balls`. Then the following tree shows the draw probabilities at each step:

```mermaid
graph TD;
    A((Start)) --> R1("R1 (10/15)");
    A --> W1("W1 (5/15)");

    R1 --> R2("R2 (9/14)");
    R1 --> W2("W2 (5/14)");

    W1 --> R2_2("R2 (10/14)");
    W1 --> W2_2("W2 (4/14)");

    R2 --> R3("R3 (8/13)");
    R2 --> W3("W3 (5/13)");

    W2 --> R3_2("R3 (9/13)");
    W2 --> W3_2("W3 (4/13)");

    R2_2 --> R3_3("R3 (9/13)");
    R2_2 --> W3_3("W3 (4/13)");

    W2_2 --> R3_4("R3 (10/13)");
    W2_2 --> W3_4("W3 (3/13)");
```

Suppose we 

## References

- [Cordani 2006 - TEACHING INDEPENDENCE AND EXCHANGEABILITY](https://iase-web.org/documents/papers/icots7/3I1_CORD.pdf)
- [Tibshirani 2023 - Conformal Predictions](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)
