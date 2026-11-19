"""Building blocks of the OpenAI-compatible adapter (`clankit.llm.openai_compat`).

"OpenAI-compatible" is a family resemblance, not a standard. Each module here owns one place
where backends disagree, so that the adapter itself stays a short pipeline:

* `profiles`   - what is known about each backend's deviations
* `wire`       - neutral messages to request bodies, including what may be replayed
* `stream`     - response chunks to stream events, tolerant of every observed chunk shape
* `arguments`  - tool-call arguments to dicts, without ever inventing content
* `text_tools` - a prompt-based tool protocol for models without usable native tool calling
* `errors`     - HTTP and in-stream failures to `ProviderError`
* `models`     - ``GET /models`` payloads to `ModelInfo`
"""
