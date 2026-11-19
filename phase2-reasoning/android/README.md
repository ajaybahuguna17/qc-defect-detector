# Person B — Independent Llama Advisor

This standalone Android project contains `:core` temporary contracts and `:advisor`.
It is separate from the official ExecuTorch LlamaDemo chat UI.

## Included
- Shared contracts: `DetectionEvent`, `DefectObservation`, `QualityDecision`, `Advisory`, `Advisor`.
- `LlamaAdvisor`: load, advise, ask, close.
- Six-class reference knowledge base, prompt builder, parser/fallbacks, in-memory cache.
- Unit tests for parsing/fallback and cache keys.
- Benchmark and sample-advisory review templates.

## Model placement
The module expects:
`<app external files>/llm/llama3_2.pte`
`<app external files>/llm/tokenizer.model`

The final app must copy/import the files into its app-specific external files directory.
Do not use `/data/local/tmp` as the production model path.

## Build
Use JDK 17 and Android SDK 35:
```bash
./gradlew :core:assembleDebug :advisor:assembleDebug
./gradlew :advisor:testDebugUnitTest
```

## Integration rules
1. Person A supplies `DetectionEvent` with defect observations, severity and the final
   decision from deterministic rules.
2. Llama only explains. It must never change ACCEPT/REWORK/REJECT.
3. The caller skips `advise()` for clean events and returns PASS from the deterministic rules.
4. When the team agrees on its final shared `:core`, move `InspectionContracts.kt` there
   and remove this temporary copy. Do not keep duplicate contracts.
5. Confirm Person A's exact class labels, severity names, area units, bounding-box schema
   and decision mapping before connecting the real detector.

## Runtime details
ExecuTorch Android 1.5.1's JNI `generate()` call blocks until native generation returns.
This advisor therefore serializes model work on a dedicated single-thread coroutine dispatcher,
not the UI thread. The implementation uses the tested `LlmModule.generate(prompt, seqLen,
callback, echo)` overload.

## Cache
In-memory key: normalized defect type | severity | decision. Cache entries are cleared on
`close()` or process exit. Repeated matching events reuse the advisory with the new event ID.

## Safety and limitations
- Severity and disposition remain deterministic.
- Missing SUMMARY/CAUSE/ACTION fields receive safe reference-based fallbacks.
- Causes are explicitly possible, not confirmed.
- The six knowledge-base entries are initial engineering notes. Verify them against cited
  steel-industry references and plant procedures before production use.
- The package has not been built on the user's Mac in this turn. Run the build and tests above
  before treating it as compile-verified.

## Remaining Person B completion checks
- Copy/import model files in the final app and test load there.
- Test all six defect classes and multiple severity/decision combinations.
- Record real model load time, first-token latency, tokens/s, uncached/cached latency and RAM.
- Prepare 15–20 reviewed sample advisories with 1–5 correctness/usefulness scores.
- Align temporary contracts with Person A's actual detector output and the final team core.
