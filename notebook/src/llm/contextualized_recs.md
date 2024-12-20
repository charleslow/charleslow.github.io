# Contextualized Recommendations

Contextualized recommendations is an emerging use case from LLMs. The idea is that we use a traditional search and recommender system to generate recommendations, and use an LLM to craft an explanation for why each recommendation is relevant to the user in a very personalized way. Multiple companies have reported that this has driven engagement and clicks up significantly.

## Spotify

[Contextualized recommendations through personalized narratives using LLMs](https://research.atspotify.com/2024/12/contextualized-recommendations-through-personalized-narratives-using-llms/)

Traditionally, spotify users use just the cover art to decide whether to engage with a new music recommendation. Spotify wants to include a short one-liner to explain why a particular item might resonate with users. For example, `Dead Rabbitts latest single is a metalcore adrenaline rush!` or `Relive U2's iconic 1993 Dublin concert with ZOO TV Live EP`. 

Spotify highlights some challenges they faced:
- Ensuring a consistent generation style and tone
- Avoiding harmful or inappropriate outputs
- Mitigating hallucinations and inaccuracies
- Understanding user preferences to deliver tailored meaningful explanations

Initial tests with zero-shot / few-shot Llama did not work too well. They adopted a human-in-the-loop approach:
- Expert editors provide "golden examples" for instruction fine-tuning
- Provide ongoing feedback to address errors in LLM output
    - Artist attribution errors
    - Tone inconsistencies
    - Factual inaccuracies

The AB tests showed that explanations containing <<meaningful details>> about the artist or music led to significantly higher user engagement.

For LLM fine-tuning, they found that Llama 3.1 8B worked well and could be trained with multiple adapters for 10 different tasks. Throughout the training process, they used MMLU benchmark as a guardrail to ensure that the model's overall ability remained intact. Spotify uses vLLM for inference.

## LinkedIn

[Our new AI powered LinkedIn](https://www.linkedin.com/pulse/celebrating-1-billion-members-our-new-ai-powered-linkedin-tomer-cohen-26vre)

LinkedIn provides AI features for premium users. When users click on a job, they can ask questions like "Am I a good fit for the job?". The LLM will respond with a short bullet-pointed explanation on:
- Whether the user is a good fit
- Details from the user's profile that make them a good fit
- Areas that the users are missing
