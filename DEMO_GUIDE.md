# Teaching this RAG workshop

## Before the session

1. Open notebook 00 with **Python (RAG Workshop)** and run the diagnostics. Verify that it says **LIVE** for a real model demonstration.
2. Run `scripts/smoke_check.py --models` once while connected to the internet to cache both local models. Run `scripts/smoke_check.py --live` to check the key/account with one short request.
3. Use the bundled fictional documents for a repeatable presentation. Open the relevant policy text and know which passage supports each answer.
4. Restart the kernel before each notebook. Run one section at a time and let the audience inspect the retrieved text, ranks, and prompt.
5. Keep notebook 09's optional judge disabled until you deliberately want extra API calls. Cached helper responses are useful for repeated demos; explain the `cached` field when discussing latency.

For the media session, rehearse playback in your actual Jupyter/VS Code frontend and warm the chosen models before presenting:

```powershell
# Use the switches for models you plan to demonstrate; these run locally on CPU.
.\.venv\Scripts\python.exe scripts/smoke_media.py --clip --asr --clap

# A short Claude vision check using the chart; this can consume API credits.
.\.venv\Scripts\python.exe scripts/smoke_media.py --vision
```

CLIP and CLAP download different model weights; ASR downloads a Whisper model. These may take substantial time and disk space on first use. They need no second paid API key. Notebook 12 enables CLIP in live mode by default. Notebook 14 defaults `RUN_ASR`, `RUN_DENSE`, and `RUN_CLAP` to `False`; notebook 15 defaults `RUN_ASR` and `RUN_CLIP` to `False`. Enable the relevant switches in each notebook's setup cell before Run All. Warm-up alone does not change those switches.

In offline rehearsal, PDF rendering, waveform playback, frame extraction, graph traversal and SQL still work. CLIP, CLAP, ASR and Claude are explicitly skipped. Describe any supplied narration as an **authored script**, not as successful speech recognition.

## A 60-minute introduction

| Minutes | Notebook | Action |
|---|---|---|
| 0–8 | 00 | Explain the difference between a language model and a searchable knowledge base; check the environment |
| 8–25 | 01 | Ask about leave, retrieve evidence, and compare the grounded answer with a prompt without evidence |
| 25–38 | 02 | Change the word window and overlap; inspect where a policy condition gets separated |
| 38–52 | 03 | Compare exact words with a paraphrase, inspect similarities, explain normalized vectors |
| 52–60 | 01 | Ask an unsupported question and discuss why abstention is useful |

## Demo questions and teaching points

| Question / change | Evidence or observation to look for |
|---|---|
| How many paid annual leave days do employees get? | `benefits_leave`: 24 for full-time employees; show the source citation |
| How many days carry forward, and until when? | 5 days; March 31. Check that a chunk includes both the number and condition |
| Can I work from home every weekday? | Standard policy allows 3 days with manager approval; extra days need an exception |
| Can a used subscription bought five days ago be refunded? | Both refund conditions matter: unused AND within 14 days |
| What does Team cost for the minimum number of users? | INR 2400 × 5 = INR 12000 monthly equivalent, annual billing; arithmetic is derived from evidence |
| Compare Team and Enterprise retention and API rates. | Needs multiple facts: 90/365 days and 600/3000 requests per minute |
| What should we do after HTTP 429? | `api_limits`: Retry-After and bounded exponential backoff with jitter; exact codes favor lexical retrieval |
| What is the parental leave allowance? | Not specified. A plausible number is a hallucination, even if it sounds reasonable |
| What about Enterprise? | Follow-up is ambiguous in isolation; notebook 08 rewrites it using conversation history |
| Can a guest retrieve the internal incident playbook? | Notebook 11 filters authorized documents before search and checks cache behavior |

## Running the advanced session

- **04:** Restart the kernel and reopen the same vector collection. Explain why a corpus/backend/model fingerprint prevents mixing incompatible vector spaces. Inspect Chroma distances: smaller is better for cosine distance.
- **05:** Inspect the fully formatted LangChain prompt before invoking the model. Compare its steps with the manual pipeline from 01.
- **06:** Write the reciprocal rank formula on screen. Demonstrate that a higher-quality final answer is not guaranteed by simply adding reranking; inspect the actual top passages.
- **07:** Show a rewrite, then check that it preserves the user's intent. Parent retrieval adds surrounding context after a narrow child match. A hypothetical HyDE answer is a search aid, not a source.
- **08:** Resolve a follow-up, reset memory, and repeat. Explain why previous assistant answers cannot serve as fresh documentary evidence.
- **09:** Tune on development questions. Freeze the choice and evaluate test questions once. Separate retrieval failure, generation failure and unknown-answer behavior. Avoid presenting keyword checks or an LLM judge as proof of truth.
- **10:** Walk through the actual `tool_use` and matching `tool_result`. Change the question so another retrieval is useful. Stop when the configured call/round budget is reached. Offline mode checks only the local dispatcher.
- **11:** Show a principal change, an evidence update, and the resulting cache miss. Use the synthetic instruction-in-document example to discuss why prompts and citation-ID checks are only partial protections.

## A separate three-hour media, graph, and SQL session

This route selects demonstrations from lessons 12–17. Allow roughly five to six hours to teach every cell and exercise in those notebooks, excluding first-time downloads.

| Minutes | Notebook | What to run and explain |
|---|---|---|
| 0–25 | 12 | Compare a chart caption with original pixels; inspect actual CLIP rankings, fuse ranks and answer with image + policy evidence |
| 25–55 | 13 | Show empty PDF text extraction, render the real pages, inspect live vision descriptions and retrieve only the refund page |
| 55–85 | 14 | Play the narration, inspect actual ASR and timestamps, replay a retrieved excerpt, then compare transcript search with CLAP sound search |
| 85–120 | 15 | Decode the MP4, extract audio and frames, map both retrieval paths to windows and answer with timestamped frames + speech evidence |
| 120–150 | 16 | Inspect sourced triples, traverse one versus two hops, preserve a negated edge and show the exact supporting passages |
| 150–180 | 17 | Inspect tables, show the Claude query plan and actual parameterized SQL, then combine computed results with billing policy |

The bundled narrated recording is 38.6 seconds. Its video contains actual slide frames and an audio track; it is not a placeholder link. Its scripts and retrieval windows are authored fixtures, while optional ASR and CLIP outputs are genuinely model-produced. Keep that distinction visible during the demonstration.

## Questions for the six new lessons

| Question / change | Evidence or observation to look for |
|---|---|
| Compare Team and Enterprise retention; what color is the taller bar? | 90/365 days and the taller purple Enterprise bar. The supplied caption contains numbers but omits colors; original pixels supply the visual fact |
| What is the Team price and minimum seat count? | Policy text: INR 2,400 per user per month, billed annually, minimum 5 users. Combine written facts with the chart in one answer |
| Why does the three-page PDF yield no extracted text? | Its pages contain images without a selectable text layer. Rendering is separate from recognizing the text in those pixels |
| What are the refund deadline and eligibility condition? | Retrieve original PDF page 3: within 14 days of purchase and an unused annual subscription. Check the cited original page |
| What are the retention periods in the recording? | Replay the retrieved waveform interval and verify ninety/three hundred and sixty-five against the displayed transcript; ASR may render numbers differently |
| Find repeated electronic beeps | CLAP embeds waveform samples, not filenames or transcripts. Inspect rankings and listen; a similarity score is not a calibrated sound-class probability |
| Which video interval shows the retention comparison? | Inspect the decoded chart frame, the overlapping speech, and the original time range. Scene labels do not enter the transcript index |
| What changes when frame retrieval is disabled? | Video ranking is transcript-only with time-based frame selection; do not call that learned visual retrieval |
| Which learning benefit should a new hire check, and what approval is needed? | Traverse onboarding → learning budget → amount/approval; inspect the source-backed INR 30,000 annual amount and manager approval |
| Does upgrading restore already expired events? | The graph edge carries explicit negation, supported by the retention policy; dropping that qualifier reverses the answer |
| How many Cedar Labs seats are unused, and what is its annual list cost before tax? | Actual SQL: 8 purchased, 6 active, 2 unused; INR 230,400 annual Team list price. Policy says taxes are additional |
| Is that SQL result the customer's amount paid? | No: list-price calculation and the order ledger are different facts; inspect the supported query and its source rows |

For **12**, remove image evidence and discuss which visual claims become unsupported. For **13**, distinguish the one-time whole-document ingestion call from the query-time selected-page answer. Switching `ANSWER_INPUT` to `"pdf"` and rerunning demonstrates native PDF input with the same retrieved subset; it adds a request unless cached.

For **14**, compare transcription errors, retrieval errors and answer errors separately. The sound-search branch does not imply that Claude heard the WAV: its answer uses timestamped transcript text. For **15**, distinguish a frame's exact timestamp, a visual scene interval and an utterance's own times. The authored visual boundaries follow the encoded frames and include intervening silence; separate narration fields preserve the original speech intervals. Inspect ASR segments that cross a visual boundary. A still frame cannot establish movement.

For **16**, the default graph is manually curated. Optional Claude extraction produces candidates for validation, not automatic truth, and the community overview is not Microsoft's full GraphRAG global-search implementation. For **17**, the model selects a reviewed query template and bound parameters. Show the read-only SQL enforcement and inspect whether the chosen intent actually matches the question; safe execution alone cannot prove relevance.

## If something fails

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Select `.venv\Scripts\python.exe` as the kernel; rerun setup if packages are missing |
| Key reported missing | Keep `CLAUDE_KEY` in the project's `.env`; restart the kernel after editing it |
| HTTP 401 | Check that the key belongs to Anthropic and is valid; do not print it in a notebook |
| HTTP 400/403/404 | Check credits, account permissions and `CLAUDE_MODEL` availability; model listing is optional in 00 |
| HTTP 400 mentions `anthropic-workspace-id` | Add `ANTHROPIC_WORKSPACE_ID=wrkspc_...` from Claude Console → Settings → Workspaces to `.env`, or use a workspace-scoped key; restart the kernel |
| HTTP 429 | Allow the bounded SDK retries; wait and reduce calls if account limits persist |
| Hugging Face download failure | Check network access and rerun the model warm-up; use explicitly labelled offline rehearsal if needed |
| CLIP/CLAP/ASR says skipped | Check the notebook's `RUN_*` switch and `RAG_OFFLINE`. Enable the branch and rerun its setup/dependent cells; no learned result is produced while skipped |
| Media model warm-up is slow or uses too much memory | Warm one model at a time, close other model kernels, and use the authored baseline for the first walkthrough. CLAP is a separate large model |
| `OFFLINE` appears unexpectedly | Check `RAG_OFFLINE` in both the process environment and `.env`; restart Jupyter after changing the parent environment |
| Results differ from the guide | Inspect actual evidence; model responses and semantic rankings can vary. Expected facts come from the corpus, not fixed generated prose |
| Empty or broken PDF extraction | Use notebook 13 to render image-only pages and perform vision ingestion, or supply verified OCR text. Plain text extraction cannot read pixels |
| Claude page-description JSON is rejected | Inspect `extraction['text']` and `stop_reason` in notebook 13. Retry or explicitly set `USE_AUTHORED_DESCRIPTIONS=True`; the notebook does not silently invent extraction output |
| Audio/video player does not render or seek | Trust the notebook if required by your frontend, inspect the saved WAV/MP4 directly, and verify the printed timestamps. Browser fragment seeking support varies |
| ASR timing validation reports clipped or discarded segments | Inspect `raw_start`, `raw_end`, and `timing_adjusted`. Playback spans are bounded to the recording; clipping does not verify the words or their alignment. Replay the interval before relying on the citation |
| Video query has weak lexical overlap | Inspect narration wording, including “retained” versus “retention”; use an appropriate query or enable the visual branch and compare actual evidence |
| SQL plan is rejected | Inspect the supported query IDs and parameter schema. Unsupported questions should stop before execution; do not bypass the validator |
| Truncated Claude answer | Inspect `stop_reason`; increase `max_tokens` for that specific demonstration |

## Capstone challenge

Replace the fictional corpus with a small, shareable knowledge base. Write at least 20 representative questions, including multi-source and unanswerable cases. Keep a held-out set. Compare lexical, dense and hybrid retrieval; choose based on measured results and latency. Add source versioning, enforce access rules before retrieval, inspect unsupported claims manually, and document the remaining limitations. The notebook code is a learning starting point, not a complete security boundary or production deployment.

For an extended capstone, add a chart, a scanned page, a short recording and a small SQL table. Preserve original file/page/time/row references. Evaluate transcription or vision extraction separately from retrieval and final answers. Add graph relationships only with supporting source quotes and qualifiers. Compare the extra complexity with a text baseline; report the cases where each method helps or fails rather than claiming that all RAG variants are covered.
