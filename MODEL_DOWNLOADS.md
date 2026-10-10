# Restoring local inference models

The recorded scripts do not automatically download missing models. src/agents.py loads model files from the models directory; a missing GGUF file makes model initialisation fail. Restore the files before starting inference. The record-only checker does not need GGUF weights.

Download exactly the pinned Q8_0 artefacts below into models/. An internet connection and approximately 4.53 GB of download/storage are required for both weights. Preserve provenance and result logs when removing weights. Fitted diagnostic predictors (.joblib) are different artefacts and are supplied in the release bundles.

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

On macOS/Linux, create models/ and download each URL with curl -L --fail --output models/FILENAME URL. Verify SHA-256 with shasum -a 256 models/FILENAME before starting an experiment. Replace FILENAME and URL with the exact values above. No new model inference or multi-gigabyte download was executed during this GitHub upload.
