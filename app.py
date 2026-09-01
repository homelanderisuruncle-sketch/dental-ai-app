import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

# =========================================================
# Groq - اختياري
# =========================================================
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# =========================================================
# gdown - لتحميل النموذج من Google Drive
# =========================================================
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


# =========================================================
# إعداد الصفحة
# =========================================================

st.set_page_config(
    page_title="Dental AI Assistant",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(37, 99, 235, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at bottom left,
                rgba(14, 165, 233, 0.08),
                transparent 30%
            ),
            #f8fafc;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a8a 55%,
            #2563eb 100%
        );
        padding: 42px;
        border-radius: 28px;
        color: white;
        margin-bottom: 30px;
        box-shadow:
            0 20px 50px rgba(15, 23, 42, 0.18);
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #dbeafe;
        line-height: 1.8;
        max-width: 800px;
    }

    .badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        color: #e0f2fe;
        font-size: 13px;
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.18);
    }

    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 24px;
        box-shadow:
            0 10px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 21px;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .card-description {
        color: #64748b;
        line-height: 1.7;
        font-size: 14px;
    }

    .diagnosis-card {
        background:
            linear-gradient(
                135deg,
                #eff6ff,
                #f8fafc
            );
        border: 1px solid #bfdbfe;
        border-radius: 22px;
        padding: 28px;
        margin-top: 20px;
    }

    .diagnosis-label {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .diagnosis-name {
        font-size: 34px;
        font-weight: 800;
        color: #1d4ed8;
        margin-bottom: 12px;
    }

    .confidence-box {
        background: white;
        border-radius: 16px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        margin-top: 15px;
    }

    .medical-warning {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        padding: 16px 18px;
        border-radius: 16px;
        line-height: 1.8;
        margin-top: 20px;
    }

    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 16px 18px;
        border-radius: 16px;
        line-height: 1.8;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding-top: 30px;
        font-size: 13px;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 18px;
        padding: 8px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        height: 48px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# إعداد النموذج
# =========================================================

CLASS_NAMES = [
    "CaS",
    "CoS",
    "Gum",
    "MC",
    "OC",
    "OLP",
]

CLASS_NAMES_AR = {
    "CaS": "CaS",
    "CoS": "CoS",
    "Gum": "مشاكل اللثة",
    "MC": "MC",
    "OC": "OC",
    "OLP": "OLP",
}


class CustomEfficientNet(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        self.model = models.efficientnet_b3(
            weights=None
        )

        in_features = (
            self.model.classifier[1].in_features
        )

        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)


# =========================================================
# معلومات Google Drive
# =========================================================

MODEL_PATH = Path("best_model.pth")

GOOGLE_DRIVE_FILE_ID = (
    "10bc8mAmX1rp1nlFqWujwL3jqC_mGw6mM"
)


# =========================================================
# تحميل النموذج
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -----------------------------------------------------
    # تحميل النموذج من Google Drive
    # -----------------------------------------------------

    if not MODEL_PATH.exists():

        if not GDOWN_AVAILABLE:

            return (
                None,
                device,
                "مكتبة gdown غير مثبتة."
            )

        try:

            url = (
                "https://drive.google.com/uc?id="
                + GOOGLE_DRIVE_FILE_ID
            )

            downloaded_file = gdown.download(
                url=url,
                output=str(MODEL_PATH),
                quiet=False,
                fuzzy=True
            )

            if (
                downloaded_file is None
                or not MODEL_PATH.exists()
            ):

                return (
                    None,
                    device,
                    "فشل تحميل النموذج من Google Drive."
                )

        except Exception as e:

            return (
                None,
                device,
                f"خطأ أثناء تحميل النموذج: {e}"
            )

    # -----------------------------------------------------
    # التحقق من حجم الملف
    # -----------------------------------------------------

    try:

        file_size = MODEL_PATH.stat().st_size

        if file_size < 1024:

            return (
                None,
                device,
                "ملف النموذج الذي تم تحميله غير صالح."
            )

    except Exception as e:

        return (
            None,
            device,
            f"تعذر التحقق من ملف النموذج: {e}"
        )

    # -----------------------------------------------------
    # إنشاء النموذج
    # -----------------------------------------------------

    try:

        model = CustomEfficientNet(
            num_classes=len(CLASS_NAMES)
        )

        # -------------------------------------------------
        # تحميل الـ weights
        # -------------------------------------------------

        try:

            checkpoint = torch.load(
                str(MODEL_PATH),
                map_location=device,
                weights_only=True
            )

        except TypeError:

            checkpoint = torch.load(
                str(MODEL_PATH),
                map_location=device
            )

        # -------------------------------------------------
        # التعامل مع أنواع Checkpoint المختلفة
        # -------------------------------------------------

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:

                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:

                state_dict = checkpoint[
                    "model_state_dict"
                ]

            else:

                state_dict = checkpoint

        else:

            return (
                None,
                device,
                "صيغة ملف best_model.pth غير مدعومة."
            )

        # -------------------------------------------------
        # إزالة module.
        # -------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):

                key = key[7:]

            cleaned_state_dict[key] = value

        # -------------------------------------------------
        # تحميل الأوزان
        # -------------------------------------------------

        model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

        model.to(device)
        model.eval()

        return model, device, None

    except Exception as e:

        return (
            None,
            device,
            f"حدث خطأ أثناء تجهيز النموذج: {e}"
        )


# =========================================================
# تجهيز النموذج
# =========================================================

with st.spinner(
    "⏳ جاري تجهيز نموذج الذكاء الاصطناعي..."
):

    model, device, model_error = load_model()


# =========================================================
# تحويل الصور
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


# =========================================================
# Header
# =========================================================

st.markdown(
    """
    <div class="hero" dir="rtl">

        <div class="badge">
            🦷 AI • Dental Image Analysis
        </div>

        <div class="hero-title">
            مساعد الأسنان الذكي
        </div>

        <div class="hero-subtitle">
            ارفع صورة للأسنان واحصل على تحليل
            بواسطة نموذج ذكاء اصطناعي مبني على
            EfficientNet-B3، مع عرض نسبة الثقة
            والاحتمالات الأعلى وتوضيح ذكي للنتيجة.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.markdown(
        "# 🦷 Dental AI"
    )

    st.markdown(
        """
        أداة مساعدة لتحليل صور الأسنان
        باستخدام الذكاء الاصطناعي.
        """
    )

    st.divider()

    st.markdown(
        "### 📋 الحالات المدعومة"
    )

    for class_name in CLASS_NAMES:

        st.markdown(
            f"• **{CLASS_NAMES_AR[class_name]}**"
        )

    st.divider()

    if device.type == "cuda":

        st.success(
            "⚡ GPU مفعل"
        )

    else:

        st.info(
            "💻 يعمل على CPU"
        )

    st.divider()

    st.caption(
        "النتائج تقديرية ولأغراض تعليمية "
        "ولا تغني عن طبيب الأسنان."
    )


# =========================================================
# خطأ النموذج
# =========================================================

if model_error:

    st.error(
        "⚠️ لم يتم تشغيل نموذج الذكاء الاصطناعي"
    )

    st.markdown(
        f"""
        <div class="card" dir="rtl">

            <div class="card-title">
                حدثت مشكلة أثناء تجهيز النموذج
            </div>

            <div class="card-description">
                {model_error}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# =========================================================
# رفع الصورة
# =========================================================

col1, col2 = st.columns(
    [1, 1],
    gap="large"
)


with col1:

    st.markdown(
        """
        <div class="card" dir="rtl">

            <div class="card-title">
                📤 ارفع صورة الأسنان
            </div>

            <div class="card-description">
                اختر صورة واضحة للأسنان بصيغة
                JPG أو JPEG أو PNG.
                <br><br>
                للحصول على نتيجة أفضل استخدم صورة
                واضحة وبإضاءة جيدة.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "اختيار الصورة",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )


# =========================================================
# إذا تم رفع صورة
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "❌ الملف المرفوع ليس صورة صالحة."
        )

        st.stop()

    with col2:

        st.markdown(
            """
            <div class="card" dir="rtl">

                <div class="card-title">
                    🖼️ معاينة الصورة
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True
        )

    st.markdown("")

    analyze = st.button(
        "🔍 تحليل الصورة الآن",
        type="primary",
        use_container_width=True
    )

    # =====================================================
    # التحليل
    # =====================================================

    if analyze:

        with st.spinner(
            "🔬 جاري تحليل الصورة..."
        ):

            try:

                input_tensor = (
                    transform(image)
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.inference_mode():

                    outputs = model(
                        input_tensor
                    )

                    probabilities = F.softmax(
                        outputs,
                        dim=1
                    )[0]

                    confidence, predicted = torch.max(
                        probabilities,
                        dim=0
                    )

                predicted_index = (
                    predicted.item()
                )

                pred_class = CLASS_NAMES[
                    predicted_index
                ]

                confidence_score = (
                    confidence.item() * 100
                )

                # =================================================
                # النتيجة
                # =================================================

                st.markdown(
                    f"""
                    <div class="diagnosis-card"
                         dir="rtl">

                        <div class="diagnosis-label">
                            النتيجة المتوقعة
                        </div>

                        <div class="diagnosis-name">
                            🦷 {CLASS_NAMES_AR[pred_class]}
                        </div>

                        <div class="confidence-box">

                            <div class="diagnosis-label">
                                ثقة النموذج
                            </div>

                            <strong>
                                {confidence_score:.2f}%
                            </strong>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(
                    min(
                        max(
                            confidence.item(),
                            0.0
                        ),
                        1.0
                    )
                )

                # =================================================
                # أعلى 3 احتمالات
                # =================================================

                st.markdown(
                    "### 📊 أعلى الاحتمالات"
                )

                top_k = min(
                    3,
                    len(CLASS_NAMES)
                )

                top_values, top_indices = (
                    torch.topk(
                        probabilities,
                        top_k
                    )
                )

                for value, index in zip(
                    top_values,
                    top_indices
                ):

                    class_name = CLASS_NAMES[
                        index.item()
                    ]

                    percentage = (
                        value.item() * 100
                    )

                    c1, c2 = st.columns(
                        [4, 1]
                    )

                    with c1:

                        st.markdown(
                            f"**{CLASS_NAMES_AR[class_name]}**"
                        )

                    with c2:

                        st.markdown(
                            f"**{percentage:.1f}%**"
                        )

                    st.progress(
                        value.item()
                    )

                # =================================================
                # Groq
                # =================================================

                st.markdown(
                    "### 🤖 التوضيح الذكي"
                )

                groq_key = None

                try:

                    if "GROQ_API_KEY" in st.secrets:

                        groq_key = st.secrets[
                            "GROQ_API_KEY"
                        ]

                except Exception:

                    groq_key = None

                if (
                    GROQ_AVAILABLE
                    and groq_key
                ):

                    try:

                        client = Groq(
                            api_key=groq_key
                        )

                        prompt = f"""
أنت مساعد تثقيفي متخصص في صحة الفم والأسنان.

قام نموذج ذكاء اصطناعي بتحليل صورة وكانت
النتيجة المتوقعة:

الحالة: {pred_class}
نسبة ثقة النموذج: {confidence_score:.1f}%

اشرح للمستخدم باللغة العربية بشكل واضح ومختصر:

- ما معنى النتيجة؟
- ما الأعراض الشائعة المرتبطة بها؟
- ما النصائح العامة للعناية بصحة الفم؟
- متى ينبغي مراجعة طبيب الأسنان؟

مهم:
لا تعتبر النتيجة تشخيصًا طبيًا نهائيًا.
لا تصف أدوية أو جرعات.
وضح أن النتيجة ناتجة عن نموذج ذكاء اصطناعي.
"""

                        response = (
                            client
                            .chat
                            .completions
                            .create(
                                model=(
                                    "llama-3.3-70b-versatile"
                                ),
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt
                                    }
                                ],
                                temperature=0.3,
                                max_tokens=800
                            )
                        )

                        answer = (
                            response
                            .choices[0]
                            .message
                            .content
                        )

                        st.markdown(
                            f"""
                            <div class="card"
                                 dir="rtl">

                                {answer}

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    except Exception as e:

                        st.warning(
                            "⚠️ تعذر تشغيل الاستشارة "
                            "الذكية حاليًا، لكن نتيجة "
                            "النموذج متاحة."
                        )

                        with st.expander(
                            "تفاصيل الخطأ"
                        ):

                            st.code(
                                str(e)
                            )

                else:

                    st.info(
                        "💡 الاستشارة الذكية غير مفعلة. "
                        "أضف GROQ_API_KEY إلى Secrets "
                        "لتفعيلها."
                    )

                # =================================================
                # تحذير طبي
                # =================================================

                st.markdown(
                    """
                    <div class="medical-warning"
                         dir="rtl">

                        ⚠️ <b>تنبيه مهم:</b><br>

                        هذه النتيجة تقديرية من نموذج
                        ذكاء اصطناعي وليست تشخيصًا طبيًا
                        نهائيًا. يجب مراجعة طبيب الأسنان
                        للحصول على تشخيص دقيق.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    "❌ حدث خطأ أثناء تحليل الصورة."
                )

                with st.expander(
                    "تفاصيل الخطأ"
                ):

                    st.code(
                        str(e)
                    )


# =========================================================
# Footer
# =========================================================

st.markdown(
    """
    <div class="footer" dir="rtl">

        🦷 Dental AI Assistant

        <br>

        AI-powered dental image analysis

        <br>

        For educational purposes only

    </div>
    """,
    unsafe_allow_html=True,
)