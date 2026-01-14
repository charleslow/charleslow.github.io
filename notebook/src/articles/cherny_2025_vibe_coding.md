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

The claude code repo has a single `CLAUDE.md` that is checked into git and everyone contributes to. Everytime claude fails on a task, or does something in a slightly undesirable way (think of all the `nit` comments), we can add a line to `CLAUDE.md` so that it does not do that again.

# 5. Use `@.claude` to add stuff to CLAUDE.md

Using the claude code github action `/install-github-action`, tagging @.claude in PR comments we can instruct claude to add stuff to CLAUDE.md. e.g. `@.claude add to CLAUDE.md to always use string literals instead of enums`.

# 6. Start sessions in plan mode

Start in plan mode `shift+tab twice`, and go back and forth. Once the plan is good, switch to auto accept edits and Claude usually 1-shots the task. A good plan is **super** important.

# 7. Use slash commands for frequent workflows

Commands are commonly executed workflows that live inside `.claude/commands/`. These can be invoked in claude chat e.g. `/commit-push-pr`. This command computes git status, and some other info before getting claude to summarize and create a PR. More details at [https://code.claude.com/docs/en/slash-commands#bash-command-execution](https://code.claude.com/docs/en/slash-commands#bash-command-execution). The `/commit-push-pr` command may be viewed [here](https://github.com/anthropics/claude-code/blob/main/.claude/commands/commit-push-pr.md)

# 8. Use sub-agents for more involved workflows

Slash commands are for short, simple workflows. Sub agents (living in `.claude/agents`) are also frequent workflows but more involved. For e.g., code simplification to simplify code after claude is done working. The code simplifier agent may be viewed [here](https://github.com/anthropics/claude-plugins-official/blob/main/plugins/code-simplifier/agents/code-simplifier.md)

# 9. Use a PostToolUse hook

Use a `PostToolUse` hook to format code after claude is done. Basically just `bun format`. We can also do `ruff format` etc. depending on the language.

# 10. Configure permissions

Do not use `--dangerously-skip-permissions`. Instead use `/permissions` and add commands that we know are safe in our environment. 

# 11. Tool Use

Use `.mcp.json` to specify the MCP servers we allow claude to connect to and use tools. 

# 12. Long Running Tasks

Use `--permission-mode=dontAsk` or `--dangerously-skip-permissions` so that claude will not get stuck. Also use hooks to ask the agent to periodically verify what is going on for monitoring.

# 13. Invest in the Verification Loop

This is the most important step. Claude needs a way to verify its work, whether it is through some unit tests, integration tests or browser usage etc. Having a verification feedback loop will 2x or 3x the quality.