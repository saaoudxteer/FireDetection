import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import timm
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_smoke_model():
    model = timm.create_model('efficientnet_b0', pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)  # ou 2 si softmax
    model.load_state_dict(torch.load("models/smoke_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


# Chargement modèles
model_fire = tf.keras.models.load_model("models/fire_model.keras")
model_env = tf.keras.models.load_model("models/environment_model.keras")
model_uncontrolled = tf.keras.models.load_model("models/uncontrolled_model.keras")

def load_torch_model(path):
    model = torch.load(path, map_location=device)
    model.eval()
    return model

model_smoke = build_smoke_model()

# YOLOv5
model_yolo = YOLO("yolov5s.pt")

# Préprocessing PyTorch
transform_torch = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Préprocessing TensorFlow
def preprocess_tf_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = preprocess_input(np.array(img))
    return np.expand_dims(img_array, axis=0)

def preprocess_torch_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return transform_torch(img).unsqueeze(0).to(device)

# === Fonctions de prédiction ===

def predict_fire(image_path):
    image = preprocess_tf_image(image_path)
    output = model_fire.predict(image)
    return float(output[0][0]) if output.shape[1] == 1 else float(output[0][1])

def predict_smoke(image_path):
    image = preprocess_torch_image(image_path)
    with torch.no_grad():
        output = model_smoke(image)
        return torch.sigmoid(output).item() if output.shape[1] == 1 else torch.softmax(output, dim=1)[:, 1].item()

def predict_uncontrolled(image_path):
    image = preprocess_tf_image(image_path)
    output = model_uncontrolled.predict(image)
    return float(output[0][0]) if output.shape[1] == 1 else float(output[0][1])

def predict_forest(image_path):
    image = preprocess_tf_image(image_path)
    output = model_env.predict(image)
    return float(output[0][0]) if output.shape[1] == 1 else float(output[0][1])

def predict_person(image_path, conf_threshold=0.4):
    results = model_yolo(image_path)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if model_yolo.names[cls_id] == "person" and conf > conf_threshold:
            return 1.0
    return 0.0
