# Restoring local inference models

The delivery launcher `run_with_models.py` downloads missing Qwen/Granite Q8_0 weights from the pinned official revisions below, verifies byte size and SHA-256, and then runs the selected unchanged research script. Valid existing weights are reused; invalid existing files are preserved and reported. Interrupted or invalid downloads are removed. Downloads happen before the experiment starts, outside recorded episode timings.

After extracting all three review archives into one directory, place this launcher in that directory and install the recorded dependencies from `requirements.txt` in a suitable Python environment. The recorded runtime used Apple Metal; other hardware/runtime combinations have not been verified here.

```bash
python run_with_models.py --model both
python run_with_models.py --model qwen src/agents.py qwen pilot
```

The first command restores both weights without running an experiment. The second restores Qwen and starts the existing Qwen pilot script. Use `--model both` for scripts that use both models. Run `python reviewer_record_check.py` directly for record analysis: it needs Python and NumPy, and no model weights.

The original scripts do not download weights when invoked directly. This launcher was added for delivery on 10 October 2026; it is not part of the recorded research code. Its download/caching/checksum behaviour was tested with small local files. No new multi-gigabyte download or inference experiment was executed during this delivery.

Both weights require an internet connection and approximately 4.53 GB of download/storage, with additional space for inference and analysis. You can remove the two named GGUF files and restore them through the launcher later. Keep provenance, result logs, data snapshots and fitted `.joblib` predictors; these predictors are different artefacts and are supplied in the archives.

## Qwen

Pinned source: https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/90862c4b9d2787eaed51d12237eafdfe7c5f6077/Qwen3-1.7B-Q8_0.gguf

File: models/Qwen3-1.7B-Q8_0.gguf

Expected SHA-256: 061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a

Expected bytes: 1834426016

## Granite

Pinned source: https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF/resolve/7cdf86ccd1f1bb3491c9b7017b033f2e51367397/granite-3.3-2b-instruct-Q8_0.gguf

File: models/granite-3.3-2b-instruct-Q8_0.gguf

Expected SHA-256: 39068a5bcb15f10487a49426d617b1b0f38a624d7dbdefe5ff6f26de82d97039

Expected bytes: 2694120096

