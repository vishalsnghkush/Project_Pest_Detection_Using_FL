# server.py
import os, io, threading, json
from flask import Flask, request, send_file, jsonify
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image

from model import MobileNetFL   # ✅ use MobileNet model
from model import SEBlock

# -------- CONFIG ----------
HOST = "0.0.0.0"
PORT = 5000

NUM_CLIENTS_PER_DATASET = {
    "color": 1,
    "segmented": 1,
    "train" :1
}

ROUNDS = 5
IMAGE_SIZE = 224   # ✅ MobileNet input size
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"C:\Users\VISHAL KUSHWAHA\OneDrive\Desktop\Federated_Learning\plantvillage dataset"
RESULTS_DIR = "server_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GLOBAL_MODEL_PATH = lambda ds: os.path.join(RESULTS_DIR, f"global_{ds}_latest.pth")
METRICS_LOG_PATH = lambda ds: os.path.join(RESULTS_DIR, f"metrics_{ds}.json")
ROUND_TRACKER_PATH = lambda ds: os.path.join(RESULTS_DIR, f"round_{ds}.txt")

DATASET_TYPES = ["color","segmented","train"]

app = Flask(__name__)

lock = threading.Lock()
updates_per_dataset = {ds: [] for ds in DATASET_TYPES}
metrics_log_per_dataset = {ds: [] for ds in DATASET_TYPES}
current_round_per_dataset = {ds: 1 for ds in DATASET_TYPES}
global_state_per_dataset = {}

# -------- helpers --------
def get_classes_list(dataset_type):
    folder = os.path.join(DATA_ROOT, dataset_type)
    if not os.path.exists(folder):
        return []
    return [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]

transforms.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])



def get_test_loader(dataset_type):
    datapath = os.path.join(DATA_ROOT, dataset_type)
    if not os.path.exists(datapath):
        return None, []
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(datapath, transform=transform)
    n = len(dataset)
    test_len = min(int(0.1 * n), 2000)
    train_len = n - test_len
    if test_len <= 0:
        return None, dataset.classes
    
    _, test_set = random_split(dataset, [train_len, test_len])
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return loader, dataset.classes

def state_dict_to_bytes(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return buf

def bytes_to_state_dict(bts):
    bio = io.BytesIO(bts)
    bio.seek(0)
    return torch.load(bio, map_location="cpu")

def fedavg(states, counts):
    total = float(sum(counts))
    agg = {}
    for k in states[0].keys():
        agg[k] = sum(states[i][k].float()*(counts[i]/total) for i in range(len(states)))
    return {k: agg[k].cpu() for k in agg}

def evaluate_state(state, dataset_type):
    loader, classes = get_test_loader(dataset_type)
    if loader is None:
        return 0.0, None, ""
    
    model = MobileNetFL(num_classes=len(classes))
    for m in model.modules():
        if isinstance(m, SEBlock):
            m.debug = True
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    




    
    y, preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            p = model(imgs).argmax(1).cpu().numpy()
            preds.extend(p.tolist())
            y.extend(labels.numpy().tolist())

    acc = accuracy_score(y, preds) * 100
    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds, target_names=classes, zero_division=0)
    return acc, cm.tolist(), report

# -------- init or load existing models --------
def init_or_load_models():
    for ds in DATASET_TYPES:
        classes = get_classes_list(ds)
        num_classes = len(classes) if classes else 38
        
        model = MobileNetFL(num_classes=num_classes).to(DEVICE)
        path = GLOBAL_MODEL_PATH(ds)

        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location="cpu"))
                print(f"✅ Loaded previous model for {ds}")
            except:
                print(f"⚠️ Failed loading saved model for {ds}, using fresh model")

        global_state_per_dataset[ds] = {k: v.cpu() for k,v in model.state_dict().items()}

        if os.path.exists(METRICS_LOG_PATH(ds)):
            metrics_log_per_dataset[ds] = json.load(open(METRICS_LOG_PATH(ds)))
        
        if os.path.exists(ROUND_TRACKER_PATH(ds)):
            current_round_per_dataset[ds] = int(open(ROUND_TRACKER_PATH(ds)).read().strip())

init_or_load_models()

# -------- Routes --------
@app.route("/get_global", methods=["GET"])
def get_global():
    ds = request.args.get("dataset", "color")
    state = global_state_per_dataset.get(ds)
    buf = state_dict_to_bytes(state)
    return send_file(buf, as_attachment=True, download_name=f"global_{ds}.pth")

@app.route("/classes")
def classes():
    ds = request.args.get("dataset", "color")
    return jsonify({"dataset": ds, "classes": get_classes_list(ds)})

@app.route("/upload_update", methods=["POST"])
def upload_update():
    ds = request.form.get("dataset")
    f = request.files["state"]
    client_id = request.form.get("client_id", "client")
    count = int(request.form.get("sample_count"))
    
    state = bytes_to_state_dict(f.read())

    with lock:
        updates_per_dataset[ds].append((state, count, client_id))
        if len(updates_per_dataset[ds]) >= NUM_CLIENTS_PER_DATASET[ds]:

            states = [u[0] for u in updates_per_dataset[ds]]
            counts = [u[1] for u in updates_per_dataset[ds]]

            agg = fedavg(states, counts)
            global_state_per_dataset[ds] = agg

            acc, _, report = evaluate_state(agg, ds)
            round_id = current_round_per_dataset[ds]

            metrics_log_per_dataset[ds].append({"round": round_id, "accuracy": acc})
            torch.save(agg, GLOBAL_MODEL_PATH(ds))

            json.dump(metrics_log_per_dataset[ds], open(METRICS_LOG_PATH(ds), "w"), indent=2)
            open(ROUND_TRACKER_PATH(ds), "w").write(str(round_id + 1))

            updates_per_dataset[ds].clear()
            current_round_per_dataset[ds] += 1

            print(f"✅ FL Round {round_id} done ({ds}) — Acc {acc:.2f}%")
            return jsonify({"status":"aggregated","round":round_id,"accuracy":acc})

    return jsonify({"status": "received"})

@app.route("/metrics")
def metrics():
    return jsonify({
        ds: {
            "current_round": current_round_per_dataset[ds],
            "metrics": metrics_log_per_dataset[ds]
        } for ds in DATASET_TYPES
    })

# ---------- GLOBAL PREDICT WITH CONFIDENCE + AUTO BEST DATASET ----------
# ================= GLOBAL CLASS LIST (Single for all datasets) =================
GLOBAL_CLASSES = []
GLOBAL_CLASS_PATH = os.path.join(DATA_ROOT, "color")  # color folder has full 38

if os.path.exists(GLOBAL_CLASS_PATH):
    GLOBAL_CLASSES = sorted([d for d in os.listdir(GLOBAL_CLASS_PATH) if os.path.isdir(os.path.join(GLOBAL_CLASS_PATH, d))])
else:
    print("❌ ERROR: Could not load global class list! Check dataset folder.")


# ============= ALWAYS USE GLOBAL CLASS COUNT FOR ALL MODELS ==============
GLOBAL_NUM_CLASSES = len(GLOBAL_CLASSES) if GLOBAL_CLASSES else 38
print(f"🌍 Using GLOBAL NUM CLASSES = {GLOBAL_NUM_CLASSES}")

# MODIFY init to ALWAYS use 38 classes
def init_or_load_models():
    for ds in DATASET_TYPES:
        model = MobileNetFL(num_classes=GLOBAL_NUM_CLASSES).to(DEVICE)
        path = GLOBAL_MODEL_PATH(ds)

        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
                print(f"✅ Loaded previous global model for {ds}")
            except:
                print(f"⚠️ Model load failed for {ds}, using fresh model.")

        global_state_per_dataset[ds] = {k: v.cpu() for k, v in model.state_dict().items()}

init_or_load_models()


# ====================== PREDICT ENDPOINT (always global classes) ======================
@app.route("/predict", methods=["POST"])
def predict():
    img = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")

    # Always use color model (or best model later)
    ds = "color"

    model = MobileNetFL(num_classes=GLOBAL_NUM_CLASSES)
    model.load_state_dict(global_state_per_dataset[ds], strict=False)
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][pred_idx])

    # ✅ REAL class name — always from GLOBAL_CLASSES
    if 0 <= pred_idx < len(GLOBAL_CLASSES):
        class_name = GLOBAL_CLASSES[pred_idx]
    else:
        class_name = f"Unknown_{pred_idx}"

    # ✅ Clean name nicely
    class_name = (
        class_name.replace("___", " — ")
                  .replace("__", " – ")
                  .replace("_", " ")
    )

    return jsonify({
        "predicted_class": pred_idx,
        "class_name": class_name,
        "dataset": "global_model",
        "confidence": confidence
    })

if __name__ == "__main__":
    print(f"🚀 Server started on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT)
