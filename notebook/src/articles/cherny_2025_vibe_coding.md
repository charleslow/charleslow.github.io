# Cherny 2025 - Vibe Coding

Boris Cherny (creator of Claude code) has a [series of tweets](https://x.com/bcherny/status/2007179833990885678) on how he uses Claude. This should be a good point of reference.

# 1. Run Claudes in 5 Terminals

Tabs are numbered 1-5. Configure system notifications to know when a claude needs input. The [link](https://code.claude.com/docs/en/terminal-config#iterm-2-system-notifications) provided is for iterm2 on Mac but probably can do something similar for linux or WSL.

# 2. Run Claudes online

Concurrently to the local terminals, he has 5-10 claudes running online at [https://claude.ai/code](https://claude.ai/code). These can be triggered either by handing off terminal sessions to the web (using `&`) or manually kicking off sessions in the browser. 
- <<Question:>> Does claude run in its own instance when running online? Doesn't that mean the context it has is different from the local context?

# 3. Use Opus 4.5 w/ Thinking all the time

Basically, use the biggest and slowest model all the time. Although it runs slower, it will complete things more reliably and with things running async, the reliability wins out in the end.

# 4. Use a CLAUDE.md 