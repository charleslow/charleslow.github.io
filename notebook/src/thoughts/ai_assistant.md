# On AI Assistants

The popularity of Openclaw and its variants show that there is much interest and fascination with having a personal AI assistant to make us more productive etc. However, after thinking about it for some time, it seems not so trivial in designing an AI assistant that actually helps (vs appearing to help).

I think my desiderata for a helpful AI assistant are:
1. <<Reliable>>. Able to autonomously and reliably complete non-trivial tasks (e.g. tasks that normally take a few hours of my time)
2. <<No brain rot>>. Do not replace or shortchange my own grokking process as a result of said offloading
3. <<Self-learning>>. Able to learn and improve over time based on my interactions with it
4. <<Tool access>>. Able to read/write/act on local data files or permissions to apps that I pass to it
5. <<Accessible>>. I am able to communicate (both input / output wise) with the assistant via convenient means (e.g. slack, telegram etc.)

Vanilla chat-based UI LLMs fulfil (1) reliable and (5) accessible. However, their self-learning and tool access are limited. 

I think these are the main pros of openclaw and its variants - the promise of (3) self-learning, creating its own tools, persisting these in a local environment, and (4) extensive permissions to interact with our files and applications (e.g. calendar, email etc.). 

These are also the easier problems to solve. The hard problems are in (1) reliable and (2) no brain rot, for which I elaborate some thoughts below.

## "Reliable" is task dependent

I think the main killer of AI assistant usage is reliability for the specific tasks that people are interested in. AI assistant reliability has passed the `[for info]` benchmark but is often not sufficient at the `[for action]` level.

For example, it is not straightforward to use AI assistants to run ML experiments. I'm trying to find the best harness for doing so (when I have time), but naively using claude code has resulted in pretty bad outcomes for me. For example, it insisted on using synthetic data instead of downloading movielens as per the paper I assigned to it - I only discovered this when the validation accuracies were abnormally high.

The problem of reliability is related to the "brain rot tax". When I am doing my own research, I gain an <<internal mental map>> as I go along that allows me to auto-correct wrong things I encounter along the way. With LLM-generated output, it feels like all-or-nothing - either we are able to trust the output 99% or we are not able to trust it at all. Suppose 80% of the report is gold and 20% is dross. We do not have the mental tools to correct the 20% because we don't know which 20% is the dross.

Another way to say a similar thing is to call it a "<<verification tax>>". Unless we have strong apriori evidence or experience to trust the output, we will have to either spend effort verifying the reliability of the output or live with a nagging feeling that something could be very wrong somewhere. The effort spent to verify the output could have been better invested in doing the research ourselves instead.

Again, my sense is that the relationship of LLM usefulness to reliability is not linear - meaning that reliability of 79% does not make it doubly useful than reliability of 40%. Instead, I think usefulness is near zero at low levels of reliability, and there is a threshold at which utility shoots up. It then grows exponentially as reliability approaches 100% before tailing off. A system that is 90% reliable is way more useful than 80%, and 95% way more than 90%. 

There is a chicken and egg problem in that we need to be an expert to design and evaluate an LLM system to have high reliability, but we often need LLM systems in areas that we are not experts. Hence the verification tax is very high for such greenfield areas that are also the most useful.

## The "brain rot" tax

Criteria 2 (no brain rot) is somewhat categorically distinct from the others, because it is not so much the tool itself but how we use it that can result in brain rot. However, I think it is worth a mention, because to me it is one of the main factors hindering effective use of AI assistants, especially for knowledge tasks.

There is a viral post going around about [tool-shaped objects](https://minutes.substack.com/p/tool-shaped-objects) that hits on this. The danger is that we often subconsciously equate lots of LLM activity with lots of productive work being done. However, this is only true if the output of the LLM activity is all we need, and the journey of getting there can be safely discarded. In reality, the journey of getting there for many knowledge tasks is just as important, and discarding the journey greatly reduces the value of the whole undertaking.

Take a research task for example. Suppose my goal is to deeply understand a particular subject. I instruct an LLM to generate a deep research report about said subject. The value of the report will not be realized until I start reading and interacting with it, asking questions, probing and trying to align my mental model with what is presented. All these activities are not captured in the report itself - they stem from my own initiative. Thus it is hard to design an AI assistant that will automatically ensure grokking happens without the user being very intentional about how they use the outputs.

Furthermore, it is not obvious apriori what form of interface best facilitates grokking, and it also varies by person and by task per person.

Thus it seems that there is always some brain rot "tax" of trading output against understanding when we use LLMs for knowledge tasks. We either recover the "tax" by separately engaging the material in intentional ways, or absorb the tax and just focus on using the output for whatever downstream purposes. Either way, there is a cognition tax that takes away from the benefits of LLM output, and often it becomes a wash.

There is probably value in designing AI assistant interfaces that explicitly mitigate brain rot through various mechanisms, such as testing the user. But I have not seen much of this yet. My own [code-stories](charleslow.github.io/code-stories) is a small attempt to find ways of encouraging user understanding of LLM material.

## Conclusion

There is no clear way forward currently. My sense is that we have to be intentional in designing specific sub-agents for each task that we want the AI assistant to do, and think carefully about the input / output medium to give us confidence that it is (1) reliable and also discourage (2) brain rot. The openclaw agent is more of an orchestrator to glue things together, but the default chat interface is not ideal for many high value tasks.
