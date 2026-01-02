import torch
import timm
import torch.nn as nn

# ---------------- RENDER-SAFE SETTINGS ----------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = torch.device("cpu")  # FORCE CPU (Render-safe)

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
_model = None  # singleton

def load_model(weight_path):
    global _model

    if _model is None:
        model = make_xception_4ch()
        model.load_state_dict(
            torch.load(weight_path, map_location="cpu")
        )

        model.eval()

        # Disable gradients (saves RAM)
        for p in model.parameters():
            p.requires_grad = False

        _model = model

    return _model
