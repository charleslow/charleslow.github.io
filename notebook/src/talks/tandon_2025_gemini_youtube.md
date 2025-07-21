# Tandon 2025 - Gemini for YouTube

[Teaching Gemini to Speak YouTube: Adapting LLMs for Video Recommendations to 2B+DAU - Devansh Tandon](https://www.youtube.com/watch?v=LxQsQ3vZDqo)

In general, recommendations is going to have higher reach than search for most consumer apps.

The personalized recommendation problem is to learn `f(user, context) = recs`.

How to rethink the whole recommendation process in terms of gemini?
- Large Recommender Model (LRM): adapting gemini for recommendation tasks
- Start with a base gemini checkpoint, adapted for youtube recommendations
- Align LRM to different tasks using adapters
    - Video retrieval (home, watch next, shorts, search, music)
    - Video ranking

To train LRM, need to develop video tokens
- Each video is one token, so we can represent watch history
- And then output some sequence of video recs

SemanticID: tokenize videos to create a language for Youtube
- Creating the atomic language for youtube videos, move away from hash based tokenizing
- `Youtube video -> extract features like title, description, transcript, audio, vidoe frames -> embedding -> RQ-VAE to quantize embedding into semantic ID`
- E.g. a semantic ID would look like `A228 B204 C196 D413 E589 ...`

Continued pre-training: get LRM to understand both English and the semantic ID "language":
- Synthetic data relationships:
    - Link text and SID, e.g. `Prompt: Video [A228 ...] has title` and the response is `Carlos Alcaraz vs Novak Djokovic`
    - Create lots of synthetic data doing this
- User behaviours: SID sequences of video watches
    - `Prompt: A user has previously watched the following vidoes: [A110, ...], [F707, ...], [C94, ...], \n <mask1>`
    - `Output: <mask1>: [B230, ...]`

After doing this kind of training, we get an LRM that can reason across both English and Youtube Video language.
- e.g. `Prompt: Video [A185, ...] is interesting to tennis fans since it is about wimbledon. Video [J110 ...] is interesting to`
- Model can respond `Output: scientists since it is about AI`

We can now perform generative retrieval with this new language. For example, we can just load context and history into the prompt, and get it to recommend new videos.
```
Prompt:
------
User: region US | 24 years female | device ANDROID | origin watch next|

Context video: channel Olympics | title WHAT a COMEBACK! Men's 400m | SemanticID_1 |

Watch history:
SID 1 Taylor Swift 100% 260.00s
SID 2 Kris Hui 40% 260.00s
```

Interestingly, compared to traditional recommenders which returns mostly men's sports videos (due to the last video being `Men's 400m`), the generative system was able to recommend `women's sprint events`.

In general:
- The LRM is able to learn very quickly, very data efficient
- Handles the toughest recs when user information is very limited
- But servicing costs are very large for YouTube scale, so alot of effort focused on reducing TPU serving cost

A simple trick to reduce serving cost is to do batch recommendations
- Build a simple video to recs table where given a seed video, get the LRM to output generic recommendations
- Prompt is something like `language {seed_lang} | duration {video_length} | age {video_age} | title {title} | channel {uploader} | {seed_sid}`
- Everyday, take `top 20M` of videos by views in `last 7days`, and do batch inference
- Use these batch recommendations to serve some users

Base Gemini is too large at YouTube scale, have to use smaller more efficient models to adapt for recsys.

> **LLM x RecSys Recipe**
> 1. <<Tokenize content>>
>   - Capture the content essence into an atomic, semantic token
> 2. <<Adapt LLM: english <> domain language>>
>   - Adapt foundation model to reason across english & new tokens (bilingual)
> 3. <<Prompt with user information>>
>   - Construct prompts with user information, activity, actions
>   - Train surface/task-specific models




