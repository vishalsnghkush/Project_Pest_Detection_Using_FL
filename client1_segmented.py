# client.py

DEBUG_MODE = True
import os, io, time, requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import MobileNetFL   # ✅ Use MobileNet Model

# ----------------------------------------------------
#                 CONFIGURATION
# ----------------------------------------------------
SERVER_URL = "http://10.181.171.47:5000/"   # ✅ no slash at end
CLIENT_ID = "client_1"
DATASET_TYPE = "segmented"   # choose: "color", "grayscale", "segmented"
LOCAL_DATA_DIR = r"C:\Users\VISHAL KUSHWAHA\OneDrive\Desktop\Federated_Learning\plantvillage dataset\segmented"

IMAGE_SIZE = 224   # MobileNet input size
BATCH_SIZE = 32
LOCAL_EPOCHS = 2
LR = 1e-3
ROUNDS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
#                 TRANSFORMS
# ----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),   # ✅ unify channels
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])


# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.Lambda(lambda img: img.convert("RGB")),
#     transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
#     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(25),
#     transforms.GaussianBlur(3),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225]),
# ])


# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225]),
#     transforms.RandomRotation(20),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
# ])


# ----------------------------------------------------
#                 LOAD LOCAL DATA
# ----------------------------------------------------
if not os.path.exists(LOCAL_DATA_DIR):
    raise SystemExit(f"[Client] ❌ Local dataset folder missing: {LOCAL_DATA_DIR}")

dataset = datasets.ImageFolder(LOCAL_DATA_DIR, transform=transform)

if len(dataset) == 0:
    raise SystemExit("[Client] ❌ No images found in local folder. Cannot train.")

train_len = int(0.9 * len(dataset))
test_len = len(dataset) - train_len
train_set, test_set = random_split(dataset, [train_len, test_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"[Client] ✅ Data loaded successfully")
print(f"         Total: {len(dataset)} | Train: {len(train_set)} | Test: {len(test_set)}")
print(f"         Device: {DEVICE}")

# ----------------------------------------------------
#      STATE SERIALIZATION HELPERS
# ----------------------------------------------------
def state_bytes_from_dict(state):
    b = io.BytesIO()
    torch.save(state, b)
    b.seek(0)
    return b

def dict_from_state_bytes(bts):
    bio = io.BytesIO(bts)
    bio.seek(0)
    return torch.load(bio, map_location="cpu")

# ----------------------------------------------------
#          DOWNLOAD GLOBAL MODEL FROM SERVER
# ----------------------------------------------------
def fetch_global():
    try:
        r = requests.get(
            f"{SERVER_URL}/get_global",
            params={"dataset": DATASET_TYPE},
            timeout=30
        )
        if r.status_code == 200:
            print(f"[Client] 🌍 Received global model from server ({DATASET_TYPE})")
            return dict_from_state_bytes(r.content)
        else:
            print("[Client] ⚠️ Server returned error while sending model:", r.text)
    except Exception as e:
        print(f"[Client] ❌ Model download failed: {e}")
    return None

# ----------------------------------------------------
#                   LOCAL TRAINING
# ----------------------------------------------------
# def train_local(global_state):
#     num_classes = len(dataset.classes)
#     model = MobileNetFL(num_classes=num_classes).to(DEVICE)

#     # Load global weights
#     if global_state is not None:
#         try:
#             model.load_state_dict(global_state, strict=False)
#         except:
#             converted = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k,v in global_state.items()}
#             model.load_state_dict(converted, strict=False)

#     opt = optim.Adam(model.parameters(), lr=LR)
#     criterion = nn.CrossEntropyLoss()

#     model.train()
#     print(f"[Client] 🚀 Training started for dataset: {DATASET_TYPE}")
#     for epoch in range(LOCAL_EPOCHS):
#         running_loss = 0.0
#         for batch_idx, (imgs, labels) in enumerate(train_loader):
#             imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#             opt.zero_grad()
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             opt.step()
#             running_loss += loss.item()

#         print(f"[Client]   🟡 Epoch {epoch+1}/{LOCAL_EPOCHS} | Loss: {running_loss/len(train_loader):.4f}")

#     # Local Evaluation
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for imgs, labels in test_loader:
#             imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#             preds = model(imgs).argmax(1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     acc = (100 * correct / total) if total > 0 else 0
#     print(f"[Client] ✅ Local model accuracy: {acc:.2f}%")

#     return {k: v.cpu() for k,v in model.state_dict().items()}, len(train_loader.dataset), acc


def train_local(global_state):
    from tqdm import tqdm  # ✅ local import to avoid breaking anything

    num_classes = len(dataset.classes)
    model = MobileNetFL(num_classes=num_classes).to(DEVICE)

    # Load global weights
    if global_state is not None:
        try:
            model.load_state_dict(global_state, strict=False)
        except:
            converted = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k,v in global_state.items()}
            model.load_state_dict(converted, strict=False)

    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(f"[Client] 🚀 Training started for dataset: {DATASET_TYPE}")

    for epoch in range(LOCAL_EPOCHS):
        running_loss = 0.0

        # ✅ Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{LOCAL_EPOCHS}", colour="green")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()

            # ✅ Show live loss in bar
            pbar.set_postfix(loss=running_loss/len(train_loader))

        print(f"[Client]   ✅ Epoch {epoch+1}/{LOCAL_EPOCHS} | Final Loss: {running_loss/len(train_loader):.4f}")

    # ✅ Local Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = (100 * correct / total) if total > 0 else 0
    print(f"[Client] 🎯 Local model accuracy: {acc:.2f}%")

    return {k: v.cpu() for k,v in model.state_dict().items()}, len(train_loader.dataset), acc


# ----------------------------------------------------
#            SEND UPDATE TO SERVER
# ----------------------------------------------------
def send_update(state, sample_count, round_idx):
    buffer = state_bytes_from_dict(state)
    files = {"state": ("state.pth", buffer.getvalue())}
    data = {
        "client_id": CLIENT_ID,
        "sample_count": str(sample_count),
        "round": str(round_idx),
        "dataset": DATASET_TYPE
    }

    try:
        r = requests.post(f"{SERVER_URL}/upload_update", files=files, data=data, timeout=90)
        print(f"[Client] 📤 Update sent to server. Response: {r.text}")
    except Exception as e:
        print(f"[Client] ❌ Failed sending update: {e}")

# ----------------------------------------------------
#            MAIN FEDERATED LEARNING LOOP
# ----------------------------------------------------
if __name__ == "__main__":
    print(f"[Client] 🌱 Federated Client Started | Dataset = {DATASET_TYPE}")
    for round_id in range(1, ROUNDS+1):
        print("\n===============================")
        print(f"        🌍 ROUND {round_id}")
        print("===============================")

        global_state = fetch_global()

        state, samples, local_acc = train_local(global_state)

        print(f"[Client] 📦 Sending state update to server | Samples used: {samples}")
        send_update(state, samples, round_id)

        print("[Client] ⏳ Waiting before next round...")
        time.sleep(3)

    print("\n🎯 Training Done! Client completed all FL rounds.")
