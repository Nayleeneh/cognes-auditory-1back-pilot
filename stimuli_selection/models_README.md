## Text-to-Speech (offline stimulus generation)

Auditory word stimuli and baseline control sounds were generated offline using the **Piper TTS** engine.

- TTS engine: **Piper**
- Language: Polish
- Voice model: `pl_PL-gosia-medium`
- Format: ONNX

The following model files are required for stimulus generation but are **not included** in the repository due to size and licensing considerations:

- `stimuli_selection/models/piper/pl_PL-gosia-medium.onnx`
- `stimuli_selection/models/piper/pl_PL-gosia-medium.onnx.json`

The model can be obtained from the official Piper repository: Hugging Face `rhasspy/piper-voices` (Polish → pl_PL → gosia → medium).