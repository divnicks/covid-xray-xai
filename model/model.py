import torch
import timm
import torch.nn as nn

# ---------------- DEVICE ----------------
DEVICE = torch.device("cpu")  # Force CPU (Render + ngrok safe)

# ---------------- MODEL DEFINITION ----------------
def make_xception_4ch():
    model = timm.create_model(
        "xception",
        pretrained=True,
        in_chans=4
    )

    in_features = model.get_classifier().in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1)
    )

    return model


# ---------------- LAZY MODEL LOADER ----------------
_model = None  # singleton (prevents re-loading & OOM)

def load_model(weight_path):
    global _model

    if _model is None:
        model = make_xception_4ch()

        model.load_state_dict(
            torch.load(weight_path, map_location=DEVICE)
        )

        model.to(DEVICE)
        model.eval()

        # Disable gradients (RAM + speed optimization)
        for p in model.parameters():
            p.requires_grad = False

        _model = model

    return _model
