import torch
import timm
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_xception_4ch():
    model = timm.create_model("xception", pretrained=True, in_chans=4)

    in_features = model.get_classifier().in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1)
    )
    return model


def load_model(weight_path):
    model = make_xception_4ch().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model
