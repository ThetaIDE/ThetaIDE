# AI Assistant

The **AI Assistant** is a Theta Hub plugin. It adds an **AI** sidebar tab where you chat with OpenAI, Anthropic, Gemini or a local Ollama model about this workspace's runs:

- why a run failed, from its logs and metrics
- how two runs differ, and which config change explains it
- what to change next, with concrete hyperparameters
- live warnings while runs train (NaN/inf, divergence, stalled learning), each with an **Ask AI** button

With your approval in a dialog, it can also edit configs (you see the diff), install Hub components, and launch or stop runs. API keys go in the OS keychain through `keyring`, or come from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`.

## Install

Open **Component Hub**, filter to **Plugins**, and install **AI Assistant**. The Hub installs its Python dependency (`keyring`) into the IDE's environment. The **AI** tab appears once installation finishes. Uninstall it the same way; your saved chats in `.thetaide/assistant/chats/` are kept.

## More

The plugin's own guide covers providers, keys, tools, run watching, cost estimates, privacy, evals and its design. It is the `README.md` shipped in the installed package (`src/plugins/app/installed/ai_assistant/README.md`) and in [theta-plugins](https://github.com/ThetaIDE/theta-plugins/tree/main/plugins/ai_assistant).
