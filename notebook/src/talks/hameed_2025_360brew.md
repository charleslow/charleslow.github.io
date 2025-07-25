# Hameed 2025 - 360Brew

[360Brew: LLM-based Personalized Ranking and Recommendation - Hamed and Maziar, LinkedIn AI](https://www.youtube.com/watch?v=U0S6CfzAY5c)

Pain points on current LinkedIn ML:
- Operational: costly low agility development lifecycle
- Quality: disjoint optimization
- Developer experience: slow to roll out changes to models one by one

Goal: build a foundational model capturing the lifetime member activity data that solves all LinkedIn matching problems
- Zero shot capability: works well out of the box for next prediction tasks
    - Measure how well the model does on new products
- In-Context learning: Learning from few examples without needing to retrain
    - How well does the model do on new users / items? [cold start]
- Follow instruction from developers / users
    - User control via prompts

## Development

Building the LLM:
- Need to convert user history into a prompt by verbalizing user information and activities
- Provide instruction on what problem we are solving
- At time of training use different styles for verbalization

Prompt looks something like:
<pre><code class="language-markdown" style="white-space: pre-wrap;">
## Instruction
You are provided a member's profile and a set of jobs, their description, and interactions that the member had with the jobs. For each past job, the member has taken one of the following actions: applied, viewed, dismissed, or did not interact. Your task is to analyze the job interaction data along with the member's profile to predict whether the member will apply, view, or dismiss a new job referred to as the "Question" job.

Note: Focus on skills, location, and years of experience more than other criteria.

## Member Profile
Current position: software engineer, current company: LinkedIn, Location: Sunnyvale, California.

## Past job interaction data
Member has applied to the following jobs: [Age: 2 days, Title: Software Engineer, Location: New York, Country: USA, Company: Meta, Description: . . . ]
Member has viewed the following jobs: [Age: 1 week, Title: Software Engineer, Location: Texas, Country: USA, Company: AMD, Description: . . . ]

## Question 1
Will the member apply to the following job: [Age: 1 day, Title: Software Engineer, Location: Seattle, Country: USA, Company: Apple, Description: . . . ]

## Question 2
Will the member apply to the following job: [Age: 5 days, Title: RF Engineer, Location: Bay Area, Country: USA, Company: Google, Description: . . . ]
</code></pre>

So in contrast to YouTube's semantic IDs, LinkedIn encodes past interactions in textual form. 

Development pipeline:
- Start with OSS model
- Continued pre-training
- Supervised Finetuning
- Alignment
- Generate Brew-XL `150B` model
- Distill to `Brew-mini`
- Prune and quantize to `Brew-mini-turbo` at `3B` parameters
    - Ablation studies show that it is critical to `first go BIG, then go small`

To make development cycle smooth, build in a lot of automation into the pipelines. Especially evaluation loop.


Three levers to improve model quality:
- More (and better data)
    - Prepare data to maximize accuracy, distribution of different type of data
- Bigger model size
- Context length
    - Longer context length means deeper user activity
    - Increasing context length initially improves performance to a certain point (around `20-30k tokens`)
    - Beyond that models don't generalize that well and performance degrades

## Tasting

Performance of model is best for cold start users. Measure relative gain over production model:
- `5 maximum activities`: `+6%`
- `10 maximum activities`: `+4%`
- `100 maximum activities`: `+2%`

Generalization to new domain. 360Brew model can generalize to out of domain tasks and surfaces and beat production models in those tasks.
- Increases team agility to roll out new features without training new model

## Serving

Three levers to improve efficiency:
- Sparsification
- Smaller model
    - Distillation from big model to small model done using SFT + KD loss
    - Gradual distillation is more effective than direct distillation, i.e. go 8B model to 6B model to 3B model etc.
    - Pruning is done layerwise, gradual pruning
- Quantization: Mix precision
    - FP8 for all weights
    - FP32 for language model head and logit processor. Important for recommendations, otherwise predictions collapse.
- Sparsification
    - Star attention (reduce attention quadratic cost)
        - Not every item needs to attend to every item
    - When scoring, we can score multiple items at the same time (sounds like `500`)
        - Need to make sure these items do not attend to each other

Q&A:
- They use 50-60 tasks out of domain to measure the effectiveness of the model in the eval loop.
- Designed custom vLLM kernels to allow multi-item scoring by modifying the attention mask