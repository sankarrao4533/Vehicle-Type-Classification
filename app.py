from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models import ResNet50_Weights, resnet50


APP_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGES = {
    "Car": APP_DIR / "car.jpg",
    "Bus": APP_DIR / "bus.jpg",
    "Truck": APP_DIR / "truck.jpg",
    "Motorcycle": APP_DIR / "motorcycle.jpg",
}
VEHICLE_GROUPS = {
    "car": {
        "sports car", "convertible", "limousine", "cab", "jeep",
        "minivan", "beach wagon", "station wagon", "racer",
    },
    "bus": {"bus", "trolleybus", "school bus", "minibus"},
    "truck": {
        "pickup", "pickup truck", "moving van", "garbage truck",
        "tow truck", "fire engine", "trailer truck", "truck",
    },
    "motorcycle": {"motorcycle", "moped", "motor scooter", "scooter"},
}
SCORE_THRESHOLD = 0.10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Vehicle Type Classifier",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": "Vehicle Type Classifier — ResNet-50 ImageNet demo"},
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #24342d;
        --muted: #69756d;
        --paper: #f5f4ed;
        --white: #fffefa;
        --rust: #ad4e32;
        --rust-dark: #873a26;
        --rust-soft: #f3e2d9;
        --amber: #b77a22;
        --amber-soft: #f2e8cd;
        --green: #386b50;
        --green-soft: #e1e9dd;
        --line: #dedfd3;
        --shadow: 0 8px 24px rgba(51, 54, 40, 0.07);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background-color: var(--paper);
        background-image: repeating-linear-gradient(
            0deg, transparent, transparent 47px, rgba(36, 52, 45, 0.025) 48px
        );
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 244, 237, 0.94);
    }

    [data-testid="stSidebar"] {
        background: #e9ece2;
        border-right: 1px solid var(--line);
        color: var(--ink);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--ink) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: 0 !important;
    }

    .masthead {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.35rem 0 1.1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--line);
    }

    .eyebrow {
        color: var(--rust);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .model-tag {
        color: var(--green);
        background: var(--green-soft);
        border: 1px solid #cbd8c5;
        border-radius: 999px;
        padding: 0.4rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .section-label {
        color: var(--muted);
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0.3rem 0 0.65rem;
    }

    .result-card {
        background: var(--white);
        border: 1px solid var(--line);
        border-left: 5px solid var(--rust);
        border-radius: 6px;
        padding: 1.35rem 1.45rem;
        box-shadow: var(--shadow);
        min-height: 205px;
    }

    .result-type {
        font-family: 'Manrope', sans-serif;
        color: var(--rust-dark);
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.1;
        text-transform: capitalize;
        margin: 0.5rem 0 0.7rem;
        overflow-wrap: anywhere;
    }

    .result-caption {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .score-number {
        color: var(--green);
        font-family: 'Manrope', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .stButton > button {
        border: 0;
        background: var(--rust);
        color: white;
        font-weight: 700;
        border-radius: 5px;
        min-height: 2.8rem;
    }

    .stButton > button:hover {
        background: var(--rust-dark);
        color: white;
    }

    [data-testid="stFileUploader"] section {
        background: var(--white);
        border: 1px dashed #bd9474;
        border-radius: 5px;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section * {
        color: var(--ink) !important;
    }

    [data-testid="stFileUploader"] button {
        background: var(--rust) !important;
        color: white !important;
        border: 0 !important;
    }

    [data-baseweb="select"] > div {
        background: var(--white) !important;
        border-color: var(--line) !important;
    }

    [data-baseweb="select"] * {
        color: var(--ink) !important;
    }

    [data-testid="stMetric"] {
        background: var(--white);
        border: 1px solid var(--line);
        border-radius: 5px;
        padding: 0.85rem 1rem;
    }

    .tip {
        background: var(--amber-soft);
        border-left: 3px solid var(--amber);
        padding: 0.75rem 0.85rem;
        color: var(--ink);
        font-size: 0.88rem;
    }

    .limitation-note {
        margin: 1.1rem 0;
        padding: 0.8rem 1rem;
        background: var(--amber-soft);
        border: 1px solid #e1d2a8;
        border-left: 4px solid var(--amber);
        border-radius: 4px;
        color: var(--ink);
        font-size: 0.9rem;
    }

    footer { visibility: hidden; }

    @media (max-width: 640px) {
        .masthead { align-items: flex-start; flex-direction: column; }
        .result-type { font-size: 2.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading ResNet-50 weights...")
def load_model():
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(DEVICE)
    model.eval()
    return model, weights.transforms(), weights.meta["categories"]


def label_to_vehicle_type(label: str) -> str | None:
    normalized_label = " ".join(label.casefold().split())
    for vehicle_type, labels in VEHICLE_GROUPS.items():
        if normalized_label in labels:
            return vehicle_type
    return None


def classify_vehicle(image: Image.Image, model, preprocess, categories):
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

    group_scores = {group: 0.0 for group in VEHICLE_GROUPS}
    matched_predictions = []
    top_values, top_indices = torch.topk(probabilities, 5)
    top_predictions = [
        (categories[index.item()], value.item())
        for value, index in zip(top_values, top_indices)
    ]

    for index, probability in enumerate(probabilities):
        label = categories[index]
        vehicle_type = label_to_vehicle_type(label)
        if vehicle_type is not None:
            score = probability.item()
            group_scores[vehicle_type] += score
            matched_predictions.append((label, vehicle_type, score))

    matched_predictions.sort(key=lambda prediction: prediction[2], reverse=True)
    predicted_type = max(group_scores, key=group_scores.get)
    vehicle_score = group_scores[predicted_type]
    if vehicle_score < SCORE_THRESHOLD:
        predicted_type = "other/uncertain"

    return predicted_type, vehicle_score, group_scores, matched_predictions[:5], top_predictions


st.markdown(
    """
    <div class="masthead">
        <div>
            <div class="eyebrow">Image classification</div>
            <h1 style="margin:0.2rem 0 0;font-size:2.2rem">Vehicle type classifier</h1>
        </div>
        <div class="model-tag">ResNet-50 · ImageNet-1K</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="section-label">Image source</div>',
        unsafe_allow_html=True
    )

    source_mode = st.radio(
        "Choose input",
        ["Upload image", "Webcam", "Project samples"],
        label_visibility="collapsed"
    )

    selected_image = None
    image_name = ""

    # -----------------------------
    # Upload Image
    # -----------------------------
    if source_mode == "Upload image":
        uploaded_file = st.file_uploader(
            "Choose a vehicle image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Use a clear vehicle image."
        )

        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_name = uploaded_file.name
            except (UnidentifiedImageError, OSError):
                st.error("This file could not be opened as an image.")

    # -----------------------------
    # Webcam
    # -----------------------------
    elif source_mode == "Webcam":
        camera_image = st.camera_input(
            "Take a picture of the vehicle"
        )

        if camera_image is not None:
            try:
                selected_image = Image.open(camera_image).convert("RGB")
                image_name = "Webcam capture"
            except (UnidentifiedImageError, OSError):
                st.error("Could not read the webcam image.")

    # -----------------------------
    # Project Samples
    # -----------------------------
    else:
        available_samples = {
            name: path
            for name, path in SAMPLE_IMAGES.items()
            if path.is_file()
        }

        if available_samples:
            sample_name = st.selectbox(
                "Choose a sample",
                list(available_samples)
            )

            image_name = available_samples[sample_name].name

            with Image.open(available_samples[sample_name]) as sample_file:
                selected_image = sample_file.convert("RGB")
        else:
            st.warning(
                "No project sample images were found next to app.py."
            )

    st.divider()

    st.markdown(
        '<div class="section-label">Runtime</div>',
        unsafe_allow_html=True
    )

    st.write(f"**Device:** `{DEVICE.type.upper()}`")

    st.caption(
        "The first run downloads the pretrained weights. "
        "Later runs reuse them from cache."
    )
if selected_image is None:
    st.markdown(
        """
        <div class="tip">
            Select a sample or upload a vehicle image. The predicted vehicle type will appear here.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

left_column, right_column = st.columns([1.05, 0.95], gap="large")

with left_column:
    st.markdown('<div class="section-label">Input image</div>', unsafe_allow_html=True)
    st.image(selected_image, caption=image_name)

with right_column:
    st.markdown('<div class="section-label">Predicted vehicle type</div>', unsafe_allow_html=True)
    try:
        model, preprocess, categories = load_model()
        with st.spinner("Analyzing image..."):
            predicted_type, vehicle_score, group_scores, mapped_predictions, top_predictions = classify_vehicle(
                selected_image, model, preprocess, categories
            )
    except Exception as error:
        st.error(f"Could not load the model or classify this image: {error}")
        st.stop()

    score_percent = vehicle_score * 100
    st.markdown(
        f"""
        <div class="result-card">
            <div class="eyebrow">This image most likely shows</div>
            <div class="result-type">{predicted_type}</div>
            <div class="result-caption">Mapped ImageNet probability mass</div>
            <div class="score-number">{score_percent:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="limitation-note">This demo maps ImageNet predictions; it is not trained specifically for vehicle categories. The mapped score is not a calibrated confidence value.</div>',
    unsafe_allow_html=True,
)

scores_column, predictions_column = st.columns(2, gap="large")
with scores_column:
    st.subheader("Vehicle group scores")
    for group, score in sorted(group_scores.items(), key=lambda item: item[1], reverse=True):
        label_column, bar_column, value_column = st.columns([1, 3, 0.8], vertical_alignment="center")
        label_column.write(group.capitalize())
        bar_column.progress(min(score, 1.0))
        value_column.caption(f"{score * 100:.2f}%")

with predictions_column:
    st.subheader("Top ImageNet predictions")
    for label, probability in top_predictions:
        st.write(f"**{label}** · {probability * 100:.2f}%")

with st.expander("Mapped vehicle labels contributing to the result"):
    if mapped_predictions:
        for label, group, probability in mapped_predictions:
            st.write(f"{label} → {group} · {probability * 100:.2f}%")
    else:
        st.write("No mapped vehicle labels contributed a measurable score.")

st.caption(
    "Model weights: TorchVision ResNet-50 IMAGENET1K_V2. "
    "A labeled vehicle dataset and fine-tuning are required for a dedicated custom classifier."
)
