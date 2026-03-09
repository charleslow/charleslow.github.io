# Picoclaw

## Concurrent Sessions

Picoclaw (sipeed/picoclaw) currently only processes one LLM message at a time. This is blocking if we want to operate on multiple sessions at a time. Here we delve into how it currently works and how we can hack it to allow concurrent sessions.

The code story is [here]()

1. <|Telegram reception|>. We start with telegram reception, which is non-blocking. Upon receiving a user telegram message, picoclaw calls `handleMessage`, which:
- Fires off some UX patterns like typing indicator, "thinking" placeholder and reactions
- Then it pushed the message onto the message bus using `PublishInbound`

2. <|Agent loop|>. There is a single `AgentLoop` that is a synchronous for loop. It calls `ConsumeInbound` to consume messages from the bus and processes them. The `processMessage` call will block until it calls the LLM, waits for response, call tools etc. This is the main blocking bottle neck.

3. <|SessionKey|>. The `BuildAgentPeerSessionKey` is the function that determines conversation isolation based on configuration
- By default, all DM messages share the same session, but this can be configured
- All group chats get their own sessions based on channel name and group id
- If we want concurrent sessions, the session key is the natural boundary to create concurrency

4. <|Session writing|>. After each `runLLMIteration`, the final contents are saved for each session. These writes are keyed by `SessionKey`, which allows two goroutines to safely run `runAgentLoop` as they read and write completely independent session files.

So if we want to hack picoclaw to support concurrent sessions (e.g. one session per telegram channel), we just need to tap on the session construct and avail it to multiple telegram channels.

## Intermediate Reasoning


