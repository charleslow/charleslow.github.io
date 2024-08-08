# Chapter 1 - Vector Spaces

## Definition 1.1 Complex Numbers $\mathbb{C}$

A complex number is an ordered pair $(a,b)$, where $a,b \in \R$, but usually written as $a + bi$.
- The set of all complex numbers is denoted as $\mathbb{C}$:

$$
\mathbb{C} = \{ a + bi: a,b \in \R \}
$$

- Addition and multiplication on $\mathbb{C}$ are defined by:
$$
\begin{align*}
    (a + bi) + (c + di) &= (a+c) + (b+d)i\\
    (a + bi)(c + di) &= (ac - bd) + (ad + bc)i
\end{align*}
$$
    Where $a, b, c, d \in \R$.

Note that $\R \subset \mathbb{C}$, since $\R = \{ a + bi : b=0, a \in \R \}$.

## Theorem 1.3 Properties of Complex Arithmetic

We may derive these properties from the definitions above.

1. Commutativity. For all $\alpha, \beta \in \mathbb{C}$,

$$
    \alpha + \beta = \beta + \alpha\\
    \alpha \beta = \beta \alpha
$$

2. Associativity. For all $\alpha, \beta, \lambda \in \mathbb{C}$,

$$
    (\alpha + \beta) + \lambda = \alpha + (\beta + \lambda)\\
    (\alpha \beta) \lambda = \alpha(\beta \lambda)
$$

3. Identities. For all $\lambda \in \mathbb{C}$,

$$
    \lambda + 0 = \lambda\\
    \lambda 1 = \lambda
$$

4. Additive Inverse is unique. For every $\alpha \in \mathbb{C}$, there exists a unique $\beta \in \mathbb{C}$ such that $\alpha + \beta = 0$.

5. Multiplicative Inverse is unique. For every $\alpha \in \mathbb{C}$ with $\alpha \neq 0$, there exists a unique $\beta \in \mathbb{C}$ such that $\alpha \beta = 1$.

6. Distributive Property. $\lambda (\alpha + \beta) = \lambda \alpha + \lambda \beta$ for all $\lambda, \alpha, \beta \in \mathbb{C}$.
