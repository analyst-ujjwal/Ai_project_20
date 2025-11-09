"""
generate_logo.py — Auto-updating Hybrid Logo Pipeline

✅ Uses Groq (LLaMA) via official SDK for smart prompt enhancement
✅ Automatically detects and replaces deprecated models
✅ Uses local Stable Diffusion Turbo for image generation
✅ Graceful fallback with detailed Streamlit log messages
"""

import os
import io
import sys
import time
from typing import List, Optional
from PIL import Image
from dotenv import load_dotenv
from prompt_enhancer import enhance_prompt as static_enhance_prompt
from utils.image_tools import post_process_logo

# --- Load .env ---
load_dotenv()

# --- Configuration ---
USE_GROQ = os.getenv("USE_GROQ", "true").lower() in ("1", "true", "yes")
USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() in ("1", "true", "yes")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama3-70b-8192")  # fallback old model name
SD_MODEL = os.getenv("SD_MODEL", "stabilityai/sd-turbo")

# --- Try imports ---
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    from diffusers import AutoPipelineForText2Image
    import torch
    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False


class GenerateLogoPipeline:
    """Main hybrid pipeline."""

    def __init__(self):
        self.device = "cuda" if DIFFUSERS_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.groq_model = self._resolve_groq_model(GROQ_LLM_MODEL)

    # --- Step 0: Auto-fix Groq model ---
    def _resolve_groq_model(self, model_name: str) -> str:
        """Check known deprecated Groq model names and replace with a working one."""
        replacements = {
            "llama3-70b-8192": "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
        }
        fixed_name = replacements.get(model_name, model_name)
        if fixed_name != model_name:
            print(f"⚙️ Replacing deprecated model '{model_name}' → '{fixed_name}'", file=sys.stderr)
        return fixed_name

    # --- Step 1: Prompt enhancement (Groq LLaMA) ---
    def _enhance_with_llama(self, prompt: str, style: str, log_callback=None) -> str:
        """Use Groq LLaMA to rewrite the logo prompt."""
        if not USE_GROQ or not GROQ_AVAILABLE or not GROQ_API_KEY:
            if log_callback:
                log_callback("⚠️ Groq SDK or API key missing — using static enhancer.")
            return static_enhance_prompt(prompt, style)

        try:
            client = Groq(api_key=GROQ_API_KEY)
            if log_callback:
                log_callback(f"Calling LLaMA ({self.groq_model}) for prompt enhancement...")

            completion = client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are a creative logo design assistant."},
                    {
                        "role": "user",
                        "content": f"Expand this idea into a detailed logo design prompt: {prompt}. Style: {style}",
                    },
                ],
                temperature=0.8,
                max_tokens=128,
            )

            enhanced = completion.choices[0].message.content.strip()
            if log_callback:
                log_callback(f"🧠 LLaMA enhanced prompt: {enhanced}")
            return enhanced

        except Exception as e:
            err = str(e)
            if "model_decommissioned" in err or "404" in err:
                if log_callback:
                    log_callback("⚠️ Model deprecated; switching to fallback model llama-3.1-8b-instant.")
                self.groq_model = "llama-3.1-8b-instant"
                return self._enhance_with_llama(prompt, style, log_callback)
            if log_callback:
                log_callback(f"⚠️ LLaMA call failed: {e}")
            print("Groq LLaMA enhancement failed:", e, file=sys.stderr)
            return static_enhance_prompt(prompt, style)

    # --- Step 2: Local Diffusion Image Generation ---
    def _ensure_local_pipe(self, log_callback=None):
        """Lazily load the local diffusion model."""
        if self.pipe is not None:
            return
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("❌ Diffusers/torch not installed.")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        if log_callback:
            log_callback(f"⚙️ Loading Stable Diffusion model: {SD_MODEL}")
        self.pipe = AutoPipelineForText2Image.from_pretrained(SD_MODEL, torch_dtype=dtype).to(self.device)

    def _generate_local(
        self, prompt: str, style: str, num_images: int, width: int, height: int, seed: int = 0, log_callback=None
    ) -> List[Image.Image]:
        """Generate logo images locally using Stable Diffusion Turbo."""
        if not USE_LOCAL:
            raise RuntimeError("Local generation disabled (USE_LOCAL=false).")
        self._ensure_local_pipe(log_callback)
        enhanced_prompt = static_enhance_prompt(prompt, style)
        if log_callback:
            log_callback(f"🎨 Generating {num_images} logo(s) locally...")

        generator = torch.Generator(device=self.device)
        if seed:
            generator.manual_seed(seed)

        results = []
        for i in range(num_images):
            img = self.pipe(
                prompt=enhanced_prompt,
                height=height,
                width=width,
                num_inference_steps=12,
                guidance_scale=6.0,
                generator=generator,
            ).images[0]
            results.append(post_process_logo(img))
            if log_callback:
                log_callback(f"✅ Generated logo {i+1}/{num_images}")
            time.sleep(0.5)
        return results

    # --- Step 3: Public generate() ---
    def generate(
        self,
        prompt: str,
        style: str = "minimal",
        num_images: int = 1,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
        use_llama: bool = True,
        log_callback=None,
    ) -> List[Image.Image]:
        """Unified entrypoint for prompt → image generation."""
        # 1. Enhance prompt
        enhanced_prompt = (
            self._enhance_with_llama(prompt, style, log_callback) if use_llama and USE_GROQ else static_enhance_prompt(prompt, style)
        )

        # 2. Local image generation
        if USE_LOCAL and DIFFUSERS_AVAILABLE:
            return self._generate_local(enhanced_prompt, style, num_images, width, height, seed, log_callback)

        raise RuntimeError("No valid backend (Groq or local) available for generation.")
