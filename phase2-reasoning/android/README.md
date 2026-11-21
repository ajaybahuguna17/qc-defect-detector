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

## Vision safety gates (Phase 1.2)
All thresholds are in `vision/.../VisionConfig.kt`. Order for one photo:

1. **Input checks before vision** (`ImageCheck.kt`, app only). Any failure -> decision **RETAKE**
   with the reason, models not run: shorter side < 64 px; mean luma < 25 or > 235; mean HSV
   saturation > 0.35 (too colourful for bare steel); variance of the 3x3 Laplacian on a 256 px
   grayscale copy < 3.0 (flat / heavily defocused). The blur threshold is deliberately low: NEU
   textures are smooth (the 12 demo examples measure 9.6 to 462), and a JVM test requires all 12
   to pass with a 3x margin.
2. **not_steel gate** (same as `phase1-vision/fusion.py`, classifier v2 only): top-1 `not_steel`
   with p >= `NOT_STEEL_THRESHOLD` (placeholder 0.50 until the Phase 1.2 run) -> "not a steel
   surface", no alert, no severity, **RETAKE**, detector boxes ignored.
3. Phase 1.1 fusion + severity, unchanged.
4. **Unsure -> HOLD** (app only, not in fusion.py): on an alert, if the classifier is below its
   threshold AND the best kept box is < 0.50, or the kept boxes cover 3+ different classes ->
   **HOLD "Unsure - manual check"**. Class, boxes and severity stay visible for the operator.
   On the 12 demo examples none is HOLD.

Llama is **never** called for RETAKE or unsure HOLD (`InspectionViewModel.needsAdvice`).
The classifier (6-class v1 or 7-class v2) is chosen from `classifier/model_io.json` +
`labels.txt` (`ClassifierSpec.kt`); `vision/build.gradle.kts` copies the classifier from
`phase1-vision/models/exported/phase1_2/classifier/` when it exists, else `phase1_1`, and the
detector always from `phase1_1`.

## Remaining Person B completion checks
- Copy/import model files in the final app and test load there.
- Test all six defect classes and multiple severity/decision combinations.
- Record real model load time, first-token latency, tokens/s, uncached/cached latency and RAM.
- Prepare 15–20 reviewed sample advisories with 1–5 correctness/usefulness scores.
- Align temporary contracts with Person A's actual detector output and the final team core.
