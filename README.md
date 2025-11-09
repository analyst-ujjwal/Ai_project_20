# AI Logo Generator


Diffusion-based text-to-logo generator. Generates clean, customizable logos from short text prompts and optional sketches. This project is intended as a starting point — plug in a diffusion checkpoint (Stable Diffusion/SDXL or Groq) and tweak presets.


## Features
- Text-to-logo generation with prompt enhancement
- Optional image-to-image refinement (sketch -> logo)
- Style presets (minimal, geometric, vintage, gradient)
- Batch generation and reproducible seeds


## Quickstart
1. Create a Python 3.10+ virtualenv
2. `pip install -r requirements.txt`
3. Place your diffusion model credentials / checkpoint as described in `generate_logo.py` comments
4. Run the Streamlit UI: `streamlit run app.py`


## Files
- `app.py` — Streamlit frontend
- `generate_logo.py` — core pipeline (uses Hugging Face diffusers by default)
- `prompt_enhancer.py` — small NLP utility to expand prompts
- `utils/` — image helpers and seed utilities
- `style_config.json` — style preset definitions


## Notes
- This repo assumes you will use a local or remote Stable Diffusion compatible checkpoint. Model-specific configuration may be required.
- Vectorization is not included but can be added (e.g., `potrace`, `autotrace`, or `DeepSVG`).