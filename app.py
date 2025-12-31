from flask import Flask, render_template, request
import os
import torch

from model.model import load_model
from model.preprocess import make_4ch_tensor
from model.xai import gradcam_simple, cam_lung_stats, save_xai_images

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = load_model("model/weights.pth")


# ---------------- PROBABILITY LABEL ----------------
def interpret_probability(prob):
    if prob < 0.30:
        return "Unlikely COVID"
    elif prob < 0.50:
        return "Possibly COVID (low confidence)"
    elif prob < 0.70:
        return "Likely COVID"
    else:
        return "Very likely COVID"


# ---------------- XAI NATURAL-LANGUAGE EXPLANATION ----------------
def explain_decision(prob, inside, outside):

    STRONG_LUNG_FOCUS = 0.64  # balanced, clinically reasonable threshold

    # -------- HIGH probability cases --------
    if prob >= 0.70:
        if inside >= STRONG_LUNG_FOCUS:
            return (
                "The model showed strong attention within the lung regions and identified patterns "
                "commonly associated with COVID-19, which supports the high probability score."
            )
        elif inside > outside:
            return (
                "The model showed greater attention within the lung regions; however, this focus was "
                "moderate rather than strong, which slightly reduces confidence despite the high probability score."
            )
        else:
            return (
                "Although the model produced a high probability score, a substantial portion of its attention "
                "was outside the lung regions, which may indicate less anatomically focused reasoning."
            )

    # -------- MODERATE probability cases --------
    elif 0.40 <= prob < 0.70:
        if inside >= STRONG_LUNG_FOCUS:
            return (
                "The model identified lung-focused patterns consistent with COVID-19; however, "
                "the overall probability remains moderate, indicating uncertainty in the assessment."
            )
        elif inside > outside:
            return (
                "The model showed limited attention within the lung regions and the probability score "
                "is moderate, suggesting a cautious interpretation of the result."
            )
        else:
            return (
                "The probability score is moderate, and the model’s attention was not primarily concentrated "
                "within the lung regions, indicating lower confidence in the assessment."
            )

    # -------- LOW probability cases --------
    else:
        return (
            "The model did not identify strong lung-focused patterns associated with COVID-19, and the "
            "low probability score suggests that COVID-19 is unlikely in this case."
        )




# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]
        filename = file.filename
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        # ---- Preprocess & inference ----
        orig_gray, lung_mask, tensor = make_4ch_tensor(path)

        with torch.no_grad():
            logit = model(tensor)
            prob = torch.sigmoid(logit).item()

        label = interpret_probability(prob)

        # ---- XAI ----
        cam = gradcam_simple(model, tensor)
        inside, outside = cam_lung_stats(cam, lung_mask)

        explanation = explain_decision(prob, inside, outside)

        # ---- Save visuals ----
        prefix = os.path.splitext(filename)[0]
        vis_paths = save_xai_images(
            orig_gray=orig_gray,
            cam=cam,
            lung_mask=lung_mask,
            save_dir=RESULT_FOLDER,
            prefix=prefix
        )

        result = {
            "prob": round(prob * 100, 2),
            "label": label,
            "inside": round(inside * 100, 1),
            "outside": round(outside * 100, 1),
            "explanation": explanation,
            "heatmap": vis_paths["heatmap"],
            "overlay": vis_paths["overlay"],
            "mask": vis_paths["mask"],
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

