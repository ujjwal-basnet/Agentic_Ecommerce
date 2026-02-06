import os, base64, requests

API_KEY = os.environ["OPENAI_API_KEY"]

person_path = "person.jpeg"
garment_path = "shirt.jpeg"
out_path = "tryon_out.png"

prompt = (
    "Virtual try-on.\n"
    "Image 1 is a person.\n"
    "Image 2 is a garment.\n"
    "Put the garment from image 2 onto the person in image 1.\n"
    "Keep face, pose, and background unchanged.\n"
    "Preserve garment color and pattern as closely as possible.\n"
)

with open(person_path, "rb") as f1, open(garment_path, "rb") as f2:
    files = [
        ("image[]", (os.path.basename(person_path), f1, "image/jpeg")),
        ("image[]", (os.path.basename(garment_path), f2, "image/jpeg")),
    ]
    data = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "size": "1024x1024",
        "quality": "low",
    }

    r = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {API_KEY}"},
        files=files,
        data=data,
        timeout=300,
    )
    r.raise_for_status()
    b64 = r.json()["data"][0]["b64_json"]

img_bytes = base64.b64decode(b64)
with open(out_path, "wb") as f:
    f.write(img_bytes)

print("saved:", out_path)