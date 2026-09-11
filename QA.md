# QA — `image-tools` branch

## Asset handling (new)
- Extracts the right assets from a screenshot — logo, hero image, feature icons (not buttons/text/whole page)
- Extracted crops are pixel-accurate and actually used in the generated page
- Uploaded an exact logo → used verbatim in the output, not redrawn
- `screenshot_preview` — agent renders its own HTML and inspects it
- `edit_images` / `remove_backgrounds` accept local asset URLs and batch independent work

## Existing functionality
- Text → code
- Video → code
- Edit / update an existing generation
- Multiple variants generate in parallel

## Across all models
- Tool images reach Gemini, OpenAI, Claude, and the Chat Completions gateways
  (they arrive as a follow-up user turn, since `tool` messages are text-only)
- Variant labels (Fast / Max / Free) show correctly

## OpenAI-compatible gateways (OpenRouter, NVIDIA NIM, Kilo, OpenCode Zen, ZenMux)
- With *only* a gateway key set, a generation runs and streams; the Settings
  dialog fields and `backend/.env` both work, and the dialog value wins
- With a first-party key also set, gateway models are not used at all
- Image input skips gateway models marked `supports_vision: false`; text input
  may use them
- Models marked `supports_tools: false` still produce HTML (no tool loop, no
  streamed preview while it writes)
- `stream_options` / `max_tokens` refusals from a strict gateway recover with
  one retry instead of failing the variant — check the backend log for
  "gateway rejected"
- OpenRouter and Kilo requests carry `HTTP-Referer` / `X-Title`
- The curated slugs still resolve: `https://openrouter.ai/api/v1/models`,
  `https://opencode.ai/zen/v1/models` and `https://zenmux.ai/api/v1/models` are
  public, so diff `backend/llm_gateways.py` against them. A retired slug surfaces
  as a variant error naming the gateway, not as a silent fallback
- Free gateways may log prompts for training (OpenRouter states this per model);
  say so in the Settings blurb of any free provider added later
- Run logs label the provider as the gateway id, and prompt reports are written
  under `prompt_report_*_openrouter_*.json`

## Running QA efficiently
- **Trust the prompt reports, not the UI.** With `PROMPT_REPORTS_ENABLED=1` + `LOGS_PATH=…`, every LLM request is logged with its tool calls, results, and final HTML — far more reliable to grep than scraping the page. Browse them at `/evals/prompt-reports`.
- **One scenario at a time — mainly for clean report attribution.** Concurrent runs interleave their prompt reports into the same folder, which is a pain to untangle (running serially also avoids piling ~4 variants each onto the providers, though I didn't actually hit a rate limit).
- **Detect "done" by the chat input returning** (the "Tell the AI what to change…" box), not by scanning page text — its placeholder isn't in `innerText`.
- **Use distinctive, deterministic fixtures** — an unmistakable logo and a clearly-structured screenshot — so you can eyeball whether the right asset was picked.
- **Assets are content-addressed** (`asset_<sha256[:24]>.png`). To prove an exact upload was used, hash the file and look for that filename in the served assets and the generated HTML.
- **Clear reports between scenarios** so each run's reports are easy to attribute.
