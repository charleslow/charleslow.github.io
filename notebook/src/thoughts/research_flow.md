# On Research Flow

Work-in-progress thoughts on how to design a personal research flow.

## Paper Reading

Paper reading can be split into 4 levels:
- <<Skim>>. Read AI-generated summaries based on a structured prompt to decide whether a paper is worth a close read. Sections may include:
    - Main Contribution
    - Main Competitors
    - Ablation Studies
    - Main Limitations
- <<Medium Dive>>. Use a CodeStories-like approach to generate an AI deep dive which is easy to browse on-the-go but still offers depth
- <<Deep Dive>>. Writeup about the paper on this blog, possibly referencing the medium dive material. Go deep into the equations and math.
- <<Replicate>>. Replicate the method and reproduce the results of the paper. This falls partially under <|experimentation flow |>.

Realistically, the quantity of papers under each category will be funnel-like:
- Skim: almost every paper in my field of interest
- Medium Dive: almost every major paper (definition of major TBD)
- Deep Dive: major papers on the topic that I am actively researching
- Replicate: subset of deep dive papers

So I need a paper reading flow like that:
- An <|automated discovery pipeline|> for finding papers to skim
    - Possibly using picoclaw, or setup a more principled approach designing something around `claude -p`
    - Allow manual addition of papers that I encounter
    - Allow steering of paper discovery process (more like this, less like this)
    - All papers should automatically go into my dropbox
- A process for assigning papers for medium dive
    - Probably using picoclaw + a PaperStories interface is sufficient
    - Expose the story on laptop + phone for browsing
    - <|TBD|> possibly a testing interface to make sure I understand
- Deep dive is on my discretion, no workflow needed here
- Replicate requires a good experimentation workflow, covered in another section

## Experimentation

More thoughts on experimentation next time. How to:
- Steer an agent to write experimentation code
- A good medium to read the code and give comments for refining
- Effective debug / full run on correct hardware (local CPU? local GPU? modal?)

Requirement: effective experimentation should be >90% steerable via conversation. But details / observability matter a lot for ML experiments. 

