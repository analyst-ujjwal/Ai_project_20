from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("./models/sd-turbo", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "futuristic tech company logo, geometric shapes, blue gradient"
image = pipe(prompt, num_inference_steps=8, guidance_scale=5.5).images[0]
image.save("test_logo.png")

print("✅ Logo generated successfully and saved as test_logo.png")
