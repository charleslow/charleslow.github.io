# On ML Experiments

On using LLMs (e.g. Claude Code) to assist with ML experiments, especially on replicating existing results from papers.

There is much value in replicating ML experiment results from papers. 
- Forces one to understand and implement the methods in the paper, no shortcuts
- Helps one understand how sensitive the results are to hyperparameters, data settings etc.
- Allows one to test out new methods or variations on the same task for an apples-to-apples comparison

Traditionally, this takes some time for each paper (days to weeks) depending on the complexity of the method, the compute cost, the expertise of the researcher who is performing the replication etc. There is also significant ambiguity in what defines a successful replication, since the paper may have inadvertently left out some implementation details, and the results might not be exactly replicable.

LLMs can make this process a lot faster, by:
- Performing requisite data cleaning
- Implementing the methods
- Monitoring the training and validation runs and fixing bugs along the way

This raises the question: what are the respective roles of the LLM and researcher in such a process? From my experience, I have tested out these loosely described paradigms:
1. <<Iterative LLM-assisted coding>>
    - Basically not too different from development *before LLMs* - the researcher codes from scratch and gets LLM to assist with each function, testing small pieces as we go along 
2. <<Spec-driven LLM development + post-hoc checking>>
    - Researcher starts with detailed specifications file, detailing as much as possible about the data, method, hyperparameters, implementation process, evaluation criteria etc.
    - LLM implements semi-autonomously, checking against the specifications
    - After each phase, researcher reviews the code to understand it, refactor it, simplify it etc.
3. <<Fully autonomous LLM>> 
    - Researcher provides the full paper to the LLM and high level specifications
    - LLM implements end to end and submits results

I find that <<approach 1>> produces the cleanest repo and understanding for the researcher. But it is also the slowest - significant time is spent debugging and doing the `code edit -> run code` loop manually. It's probably the approach that most researchers are using currently, to varying degrees of autonomy given to the LLM.

I could not get <<approach 3>> to work. I tried running Claude Code in `--dangerously-allow-permissions` mode on runpod, and give it almost full autonomy in implementing and running the code. Some frustrating points:
- <<LLM kept using synthetic data>> even for the full run, even when explicitly spelled out that synthetic data is only for dev runs. When asked why it was using synthetic data, it said it failed to download the data. But managed to download the data on the very next instruction.
- <<LLM has no common sense>>. The paper detailed its method of generating synthetic queries on movielens using an LLM (since movielens only has interaction data) to train a search biencoder. Claude decided to generate synthetic queries of the form "What movie is <title>?", despite being told to use an LLM to generate from the movie text. Such hard-coded queries obviously led to very high accuracies of `> 99%`, but Claude did not know to terminate and investigate.
- <<LLM did not create meaningful evals>>. It mostly monitored train and validation loss, and even these were not very meaningful (it decided to sample only `500` points for validation). The researcher needs to be explicit about specifying these.

Some reflections on testing out <<approach 3>>:
- <<Verification is hard in ML tasks>>. I tried providing the accuracy results from the paper to the LLM, asking it to benchmark against those figures to see if it is on the right track. The LLM was way off but did not know how to troubleshoot such problems. Normal test driven development also does not work for ML Tasks.
- <<Difficult to debug in fully autonomous setting>>. ML tasks are very sensitive to how the data is processed, hyperparameters etc. Even a small decision like left or right padding can significantly change results. Without diving into the code, I had no means of forming hypotheses about what was messing up the performance. LLMs are not well trained in this task of playing detective, and interrogating it did not yield much fruit.
- <<Unpleasant to understand code>>. Giving LLM full autonomy produces a lot of code that is intimidating to try to grok. I can probably do it with LLM assistance, but it produces the same daunting feeling as reviewing a large pull request. It also feels like unnecessary work as it entails spending time trying to understand lots of glue code.

Which leads me to <<approach 2>>. This is the approach I am currently testing out. The hope is that <<spec-driven development is a good middle ground>>. The researcher gets to use the specification to determine how far we want the LLM to sprint in each run, before we apply manual verification and understanding. This minimizes the chance of LLMs running completely off point and also makes it more palatable to understand a small piece of code each time. 

Front-loading the development of specifications also forces the researcher to adopt systems-level thinking and design for the whole experiment. This hopefully helps us move up a level of abstraction and move along faster.

The main difficulties of <<approach 2>> a-priori to me are:
- Designing the right places to pause for manual verification and understanding
- Getting the LLM to follow our specs closely and stop at the correct checkpoints

Will test it out and see how it goes!


