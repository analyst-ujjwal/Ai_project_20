"""
Streamlit app for AI Logo Generator (Groq LLaMA + Groq Flux image + local fallback)

Features:
- Optionally use LLaMA to rewrite prompt before image generation.
- Uses Groq image models (flux-dev/flux-pro) by default.
- Shows live logs in the UI.
- Retry-safe Groq calls with rate-limiting.
- "Download all as ZIP" + per-image download buttons.
"""

import streamlit as st
import os
import sys
import io
import zipfile
from PIL import Image
from generate_logo import GenerateLogoPipeline

# Page config
st.set_page_config(page_title="AI Logo Generator", layout="centered")
st.title("🎨 AI Logo Generator — Groq + Local Hybrid")

# Sidebar controls
st.sidebar.header("⚙️ Generation Settings")
style = st.sidebar.selectbox("Style preset", ["minimal", "geometric", "gradient", "vintage"])
num_images = st.sidebar.slider("Batch size", 1, 6, 2)
use_llama = st.sidebar.checkbox("Use LLaMA to enhance prompt (creative)", value=True)
use_local_checkbox = st.sidebar.checkbox("Allow local diffusers fallback if Groq fails", value=False)
seed = st.sidebar.number_input("Seed (0 for random)", min_value=0, value=0)
width = st.sidebar.selectbox("Width", [512, 640, 768])
height = st.sidebar.selectbox("Height", [512, 640, 768])

prompt = st.text_input("Brand prompt", "modern tech company logo with blue gradient and geometric shapes")

# Info
st.caption("Backend: Groq image models (flux-dev/flux-pro) by default. If Groq fails and you enabled local fallback, the app will try a local diffusers model.")

# Live log container
log_box = st.empty()
_log_lines = []

def log(msg: str):
    """Append and display last N log lines in the UI plus print to stderr for terminal logs."""
    _log_lines.append(msg)
    # keep only last 20 lines
    display = "\n".join(_log_lines[-20:])
    log_box.code(display, language="bash")
    print(msg, file=sys.stderr)

# Cached pipeline
@st.cache_resource
def get_pipeline():
    return GenerateLogoPipeline()

pipeline = get_pipeline()

# Generation section
if st.button("🚀 Generate"):
    # Ensure pipeline preference reflects UI fallback checkbox
    pipeline.prefer_groq = True  # we prefer Groq by default
    # If user allows local fallback, set USE_LOCAL at runtime
    # (Note: actual env variable USE_LOCAL controls behavior in generate_logo.py for safety)
    # We'll inform user to set USE_LOCAL env var if they want persistent local fallback.
    if use_local_checkbox:
        log("Note: local fallback requested in UI — ensure USE_LOCAL=true in environment for persistent fallback.")
    _log_lines.clear()
    with st.spinner("Generating — contacting Groq and/or local model..."):
        try:
            imgs = pipeline.generate(
                prompt=prompt,
                style=style,
                num_images=num_images,
                seed=seed,
                width=width,
                height=height,
                use_llama=use_llama,
                log_callback=log,
            )
        except Exception as e:
            st.error(f"Generation raised an exception: {e}")
            log(f"Exception during generation: {e}")
            imgs = []

    # Display results and downloads
    if imgs and len(imgs) > 0:
        st.success(f"✅ Generated {len(imgs)} image(s).")
        # Show images in rows of up to 3 columns
        per_row = 3
        total = len(imgs)
        idx = 0

        # Prepare ZIP buffer
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w") as zf:
            for row_start in range(0, total, per_row):
                row_imgs = imgs[row_start : row_start + per_row]
                cols = st.columns(len(row_imgs))
                for col, im in zip(cols, row_imgs):
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    buf.seek(0)
                    # Display image (future-proof width parameter)
                    col.image(buf, caption=f"Logo {idx+1}", width="stretch")
                    col.download_button(
                        label=f"⬇️ Download Logo {idx+1}",
                        data=buf.getvalue(),
                        file_name=f"logo_{idx+1}.png",
                        mime="image/png",
                        key=f"dl_{idx}"
                    )
                    # Add to ZIP
                    zf.writestr(f"logo_{idx+1}.png", buf.getvalue())
                    idx += 1

        # Show ZIP download
        zip_buffer.seek(0)
        st.download_button(
            label="🗜️ Download All Logos (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="logos.zip",
            mime="application/zip",
            key="download_all_zip",
        )
    else:
        st.warning("⚠️ No logos generated. Check the logs above for Groq responses and errors.")
        log("No images were returned by any backend. Check GROQ_API_KEY, model names, or local diffusers availability.")

st.markdown("---")
st.caption("Logs appear above; check the terminal where Streamlit was launched for full stderr traces.")
