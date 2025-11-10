import torch
from PIL import Image
from torchvision import transforms as T
from parseq.models.parseq import PARSeq

# Load model
model = PARSeq()
model.load_state_dict(torch.load("parseq_custom.pt", map_location="cpu"))
model.eval()

transform = T.Compose([
    T.Resize((32, 256)),
    T.ToTensor(),
    T.Normalize(0.5, 0.5),
])

img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(img)
    text = model.charset.decode(out)

print("Pred:", text)
