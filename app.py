import os
import io
import warnings

import gdown
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3

# ============================================================
# Groq AI Agent Integration
# ============================================================
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False


# ============================================================
# Optional Grad-CAM
# ============================================================
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Dental AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Class Names
# ============================================================
CLASS_NAMES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Tooth Discoloration",
    "Ulcer",
]


# ============================================================
# Google Drive Model
# ============================================================
MODEL_PATH = "best_model.pth"

FILE_ID = "10bc8mAmX1rp1nlFqWujwL3jqC_mGw6mM"

MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"


# ============================================================
# Custom EfficientNet-B3
# ============================================================
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.model = efficientnet_b3(weights=None)

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# Download Model
# ============================================================
def download_model():
    if os.path.exists(MODEL_PATH):
        return True

    try:
        with st.spinner("جاري تحميل نموذج الذكاء الاصطناعي..."):

            downloaded = gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=False
            )

        if downloaded and os.path.exists(MODEL_PATH):
            return True

        return False

    except Exception as e:
        st.error("فشل تحميل النموذج من Google Drive.")
        st.exception(e)
        return False


# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():

    if not download_model():
        return None, None

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CustomEfficientNet(
        num_classes=len(CLASS_NAMES)
    )

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device
        )

        # ----------------------------------------------------
        # Handle different checkpoint formats
        # ----------------------------------------------------
        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            else:
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        # ----------------------------------------------------
        # Remove "module." if model was trained with DataParallel
        # ----------------------------------------------------
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):
                new_key = key[len("module."):]
            else:
                new_key = key

            cleaned_state_dict[new_key] = value

        model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

        model.to(device)
        model.eval()

        return model, device

    except RuntimeError as e:

        st.error(
            "حدث خطأ أثناء تحميل أوزان النموذج."
        )

        st.error(
            "تأكد أن best_model.pth هو نموذج EfficientNet-B3 "
            "المدرب على نفس الـ 6 classes."
        )

        st.exception(e)

        return None, None

    except Exception as e:

        st.error("تعذر تحميل النموذج.")
        st.exception(e)

        return None, None


# ============================================================
# Image Preprocessing
# ============================================================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# Prediction
# ============================================================
def predict_image(model, device, image):

    input_tensor = preprocess(image).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    pred_idx = int(predicted.item())

    pred_class = CLASS_NAMES[pred_idx]

    conf_score = float(confidence.item())

    all_probabilities = probabilities[0].detach().cpu().numpy()

    return (
        pred_idx,
        pred_class,
        conf_score,
        all_probabilities,
        input_tensor,
    )


# ============================================================
# Grad-CAM
# ============================================================
def generate_gradcam(
    model,
    input_tensor,
    pred_idx,
    original_image
):

    if not GRADCAM_AVAILABLE:
        return None

    try:

        # ----------------------------------------------------
        # EfficientNet-B3 final convolutional feature layer
        # ----------------------------------------------------
        target_layers = [
            model.model.features[-1]
        ]

        # ----------------------------------------------------
        # Create GradCAM
        #
        # Newer pytorch-grad-cam versions:
        # GradCAM(model=..., target_layers=...)
        # ----------------------------------------------------
        try:

            cam = GradCAM(
                model=model,
                target_layers=target_layers
            )

        except TypeError:

            # Compatibility fallback
            cam = GradCAM(
                model,
                target_layers
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # targets MUST be passed to cam(...)
        # ----------------------------------------------------
        targets = [
            ClassifierOutputTarget(pred_idx)
        ]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        # ----------------------------------------------------
        # Convert original image to 224x224
        # ----------------------------------------------------
        resized_image = original_image.resize(
            (224, 224)
        ).convert("RGB")

        rgb_image = np.asarray(
            resized_image
        ).astype(np.float32) / 255.0

        # ----------------------------------------------------
        # Create Grad-CAM overlay
        # ----------------------------------------------------
        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------
        try:
            cam.activations_and_grads.release()
        except Exception:
            pass

        return Image.fromarray(visualization)

    except Exception as e:

        st.warning(
            "تعذر إنشاء Grad-CAM لهذه الصورة."
        )

        return None


# ============================================================
# Confidence Label
# ============================================================
def confidence_level(confidence):

    if confidence >= 0.90:
        return "Very High"

    if confidence >= 0.75:
        return "High"

    if confidence >= 0.50:
        return "Moderate"

    return "Low"


# ============================================================
# Header
# ============================================================
st.title("🦷 Dental AI")

st.write(
    "نظام ذكاء اصطناعي لتحليل صور الأسنان "
    "وتصنيف الحالة المحتملة."
)

st.divider()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:

    st.header("🦷 Dental AI")

    st.write(
        "EfficientNet-B3"
    )

    st.write(
        f"عدد الفئات: **{len(CLASS_NAMES)}**"
    )

    st.divider()

    st.subheader("الفئات")

    for i, class_name in enumerate(CLASS_NAMES, start=1):

        st.write(
            f"{i}. {class_name}"
        )

    st.divider()

    device_text = (
        "GPU (CUDA)"
        if torch.cuda.is_available()
        else "CPU"
    )

    st.caption(
        f"Device: {device_text}"
    )


# ============================================================
# Load Model
# ============================================================
model, device = load_model()


if model is None:

    st.error(
        "النموذج غير متاح. "
        "تحقق من ملف best_model.pth وGoogle Drive."
    )

    st.stop()


# ============================================================
# File Upload
# ============================================================
uploaded_file = st.file_uploader(
    "📷 ارفع صورة الأسنان",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    help="ارفع صورة واضحة للأسنان للحصول على أفضل نتيجة."
)


# ============================================================
# Main Application
# ============================================================
if uploaded_file is not None:

    try:

        image_bytes = uploaded_file.getvalue()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        st.error(
            "الملف المرفوع ليس صورة صالحة."
        )

        st.stop()

    st.divider()

    # ========================================================
    # Image Preview
    # ========================================================
    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        st.subheader("📷 الصورة الأصلية")

        st.image(
            image,
            width="stretch"
        )

    # ========================================================
    # Prediction
    # ========================================================
    with st.spinner("جاري تحليل الصورة..."):

        (
            pred_idx,
            pred_class,
            conf_score,
            all_probabilities,
            input_tensor,
        ) = predict_image(
            model,
            device,
            image
        )

    with col2:

        st.subheader("🔍 النتيجة")

        st.metric(
            "التصنيف المتوقع",
            pred_class
        )

        st.metric(
            "Confidence",
            f"{conf_score * 100:.2f}%"
        )

        st.write(
            f"مستوى الثقة: **{confidence_level(conf_score)}**"
        )

        st.progress(
            min(max(conf_score, 0.0), 1.0)
        )

    st.divider()

    # ========================================================
    # Class Probabilities
    # ========================================================
    st.subheader("📊 احتمالات جميع الفئات")

    probability_columns = st.columns(3)

    for i, class_name in enumerate(CLASS_NAMES):

        probability = float(
            all_probabilities[i]
        )

        with probability_columns[i % 3]:

            st.metric(
                class_name,
                f"{probability * 100:.2f}%"
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

    st.divider()

    # ========================================================
    # Grad-CAM
    # ========================================================
    st.subheader("🔥 Grad-CAM")

    if GRADCAM_AVAILABLE:

        with st.spinner(
            "جاري إنشاء خريطة Grad-CAM..."
        ):

            cam_image = generate_gradcam(
                model=model,
                input_tensor=input_tensor,
                pred_idx=pred_idx,
                original_image=image
            )

        if cam_image is not None:

            st.image(
                cam_image,
                caption=(
                    f"Grad-CAM — {pred_class}"
                ),
                width="stretch"
            )

            st.caption(
                "توضح الخريطة المناطق التي ساهمت "
                "أكثر في قرار النموذج."
            )

    else:

        st.info(
            "Grad-CAM غير متاح. "
            "ثبّت مكتبة grad-cam من requirements.txt."
        )

    st.divider()

    # ========================================================
    # Download Grad-CAM
    # ========================================================
    if (
        GRADCAM_AVAILABLE
        and cam_image is not None
    ):

        buffer = io.BytesIO()

        cam_image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ تحميل Grad-CAM",
            data=buffer.getvalue(),
            file_name="dental_gradcam.png",
            mime="image/png",
            width="stretch"
        )

    st.divider()

    # ========================================================
    # AI Medical Assistant (Groq Agent)
    # ========================================================
    st.subheader("🤖 المساعد الطبي للذكاء الاصطناعي")

    if GROQ_AVAILABLE and "GROQ_API_KEY" in st.secrets:
        try:
            with st.spinner("جاري توليد التقرير والنصائح الطبية..."):
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                
                system_prompt = (
                    "You are an expert dental AI medical assistant. Provide clear, compassionate, "
                    "and informative explanations to the user in Arabic."
                )
                user_prompt = (
                    f"تم تشخيص الصورة بالحالة التالية:  {pred_class}  بنسبة ثقة {conf_score * 100:.2f}%.\n"
                    f"يرجى كتابة تقرير طبي مبسط يشمل:\n"
                    f"1. شرح أسباب هذه الحالة بشكل مبسط.\n"
                    f"2. الأعراض المعتادة المرتبطة بها.\n"
                    f"3. نصائح وإرشادات هامة للعناية والوقاية.\n"
                    f"4. التوصية بما يجب فعله عند زيارة طبيب الأسنان."
                )

                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                st.write(response.choices[0].message.content)

        except Exception as e:
            st.error("حدث خطأ أثناء التواصل مع Groq AI Agent.")
            st.exception(e)

    elif not GROQ_AVAILABLE:
        st.info("مكتبة groq غير مثبتة في النظام.")
    else:
        st.info("لم يتم إعداد GROQ_API_KEY في Streamlit Secrets.")

    st.divider()

    # ========================================================
    # Medical Disclaimer
    # ========================================================
    st.warning(
        "⚠️ هذه النتيجة هي مساعدة آلية وليست تشخيصًا "
        "طبيًا نهائيًا. يجب تأكيد الحالة بواسطة طبيب أسنان مختص."
    )


# ============================================================
# No Image
# ============================================================
else:

    st.info(
        "👆 ارفع صورة أسنان من الأعلى للبدء."
    )

    st.subheader("الفئات التي يستطيع النموذج تصنيفها")

    cols = st.columns(3)

    for i, class_name in enumerate(CLASS_NAMES):

        with cols[i % 3]:

            st.info(
                f"**{class_name}**"
            )


# ============================================================
# Footer
# ============================================================
st.divider()

st.caption(
    "Dental AI • EfficientNet-B3 • "
    "AI-assisted dental image classification"
)