import os
import io
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from groq import Groq

import os
import gdown

model_path = "best_model.pth"
if not os.path.exists(model_path):
    # استبدل YOUR_FILE_ID بالـ ID الخاص بملفك على Google Drive
    file_id = "10bc8mAmX1rp1nlFqWujwL3jqC_mGw6mM"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)


# 1. إعدادات الصفحة
st.set_page_config(page_title="Dental AI Assistant", layout="wide")
st.title("🦷 نظام التشخيص الطبي للأسنان والمساعد الذكي")

# 2. تعريف نموذج EfficientNet-B3
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        # تحميل النموذج الأساسي
        self.model = efficientnet_b3(weights=None)
        # تعديل الطبقة الأخيرة لتتناسب مع عدد الفئات لديك
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    class_names = ['class1', 'class2', 'class3']  # ضع أسماء الفئات الخاصة بك هنا
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # تنزيل النموذج عبر gdown إذا لم يكن موجوداً
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        file_id = "10bc8mAmX1rp1nlFqWujwL3jqC_mGw6mM"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    # إنشاء النموذج وتمرير عدد الفئات
    model = CustomEfficientNet(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device, class_names

# تحويلات الصور
from torchvision import transforms
from PIL import Image
# تعريف التحويلات المطلوبة للنموذج
transform = transforms.Compose([
    transforms.Resize((224, 224)), # أو الحجم الذي قمت بتدريب EfficientNet عليه مثل (300, 300)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. إعداد Groq AI Agent
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None

# --- الواجهة والتفاعل ---
col1, col2 = st.columns([1, 1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with col1:
    st.subheader("1. رفع صورة الأسنان")
    uploaded_file = st.file_uploader("اختر صورة للتشخيص...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # التنبؤ
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        pred_idx = predicted.item()
        pred_class = class_names[pred_idx]
        conf_score = confidence.item() * 100
        
        # Grad-CAM
        target_layer = [model.model.features[-1]]
        grad_cam = GradCAM(model=model, target_layers=target_layer)
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)[0]
        
        img_np = np.array(image.resize((224, 224))) / 255.0
        cam_vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # حفظ النتائج في Session State
        st.session_state['pred_class'] = pred_class
        st.session_state['conf_score'] = conf_score
        
        st.success(f"التشخيص المتوقع: {pred_class} ({conf_score:.2f}%)")
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.image(image, caption="الصورة الأصلية", use_column_width=True)
        with sub_col2:
            st.image(cam_vis, caption="Grad-CAM Heatmap", use_column_width=True)

with col2:
    st.subheader("2. استشارة المساعد الطبي الذكي (AI Agent)")
    if 'pred_class' in st.session_state:
        st.info(f"الحالة المحددة للتحليل: {st.session_state['pred_class']}")
        user_query = st.text_area("اطرح سؤالك أو استفسارك حول الحالة:")
        
        if st.button("إرسال الاستفسار"):
            if not client:
                st.error("يرجى ضبط مفتاح GROQ_API_KEY في إعدادات البيئة (Secrets).")
            elif user_query.strip() == "":
                st.warning("يرجى كتابة سؤال أولاً.")
            else:
                with st.spinner("جاري تحليل الاستفسار وإعداد الرد الطبي عبر Groq..."):
                    system_prompt = (
                        "أنت مساعد طبي ذكي متخصص في طب وجراحة الأسنان.\n"
                        "اشرح حالة المريض المكتشفة بأسلوب مبسط، وضح الأسباب والأعراض، "
                        "وقدم نصائح للعناية بها مع الإجابة على سؤاله بدقة. "
                        "شدد دائماً على ضرورة زيارة الطبيب للكشف السريري."
                    )
                    user_msg = (
                        f"التشخيص المكتشف: {st.session_state['pred_class']}\n"
                        f"نسبة التأكد: {st.session_state['conf_score']:.2f}%\n"
                        f"سؤال المريض: {user_query}"
                    )
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg}
                        ]
                    )
                    st.write("---")
                    st.markdown("### 💬 إجابة المساعد الطبي:")
                    st.write(response.choices[0].message.content)
    else:
        st.write("قم برفع صورة أفقياً في القسم الأيسر للبدء في الاستشارة.")