
import streamlit as st
import cv2
import numpy as np
# from PIL import Image
import io


st.set_page_config(page_title="Leaf DGCI Classroom", layout="centered")

# Header
st.title("🌿 Leaf Health Teaching Tool")
st.markdown("""
Learn how to:
1. Upload and visualize a leaf photo  
2. Compute **DGCI** and predict chlorophyll  
3. Take a quiz on plant health concepts  
""")

# SECTION 1: Upload + DGCI computation
st.header("1. Upload Leaf Image")
uploaded = st.file_uploader("Upload a leaf image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded:
    # img = Image.open(uploaded).convert("RGB")
    # st.image(img, caption="Uploaded Image", use_container_width=True)
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # arr = np.array(img)  # img passed earlier from PIL
    # hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
   
    st.subheader("1. Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # DGCI pipeline
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = h.astype(np.float32) * 2
    s = s.astype(np.float32) / 255.0
    v = v.astype(np.float32) / 255.0
    dgci = np.clip(((h - 60) / 60 + (1 - s) + (1 - v)) / 3, 0, 1)

    heatmap = cv2.applyColorMap((dgci * 255).astype(np.uint8), cv2.COLORMAP_JET)

    lower_hue = np.array([20, 30, 30])
    upper_hue = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        st.warning("No leaf detected.")
    else:
        leaf_contour = max(contours, key=cv2.contourArea)
        mask_main = np.zeros_like(mask_clean)
        cv2.drawContours(mask_main, [leaf_contour], -1, 255, thickness=-1)

        contour_img = img.copy()
        cv2.drawContours(contour_img, [leaf_contour], -1, (255, 0, 0), 2)

        masked_dgci = np.zeros_like(dgci)
        masked_dgci[mask_main > 0] = dgci[mask_main > 0]
        final_heat = cv2.applyColorMap((masked_dgci * 255).astype(np.uint8), cv2.COLORMAP_JET)
        leaf_only = cv2.bitwise_and(final_heat, final_heat, mask=mask_main)

        mean_dgci = masked_dgci[mask_main > 0].mean()
        st.metric("Mean DGCI", f"{mean_dgci:.3f}")

        # Display 4 images
        cols = st.columns(2)
        with cols[0]:
            st.subheader("2. Leaf Mask")
            st.image(mask_clean, use_container_width=True, clamp=True)
        with cols[1]:
            st.subheader("3. Leaf Contour")
            st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with cols[0]:
            st.subheader("4. DGCI Heatmap (Full)")
            st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_container_width=True)
        with cols[1]:
            st.subheader("5. DGCI Heatmap on Leaf")
            st.image(cv2.cvtColor(leaf_only, cv2.COLOR_BGR2RGB), use_container_width=True)




# SECTION 2: Teaching Content
st.header("2. How It Works")

# SECTION 3: Quiz
st.header("3. Quick Quiz")
if "quiz_done" not in st.session_state:
    st.session_state.quiz_done = False

if not st.session_state.quiz_done:
    q1 = st.radio("1. A **lower** DGCI value indicates a leaf that is:", 
                  ["Deep green (healthy)", "Yellow or chlorotic", "Bright red/purple"])
    q2 = st.radio("2. DGCI uses images converted to:", 
                  ["RGB color space", "HSV color space", "L*a*b color space"])
    if st.button("Submit Quiz"):
        correct = 0
        st.session_state.quiz_done = True
        if q1 == "Yellow or chlorotic": correct += 1
        if q2 == "HSV color space": correct += 1
        st.success(f"You scored {correct}/2!")
else:
    st.write("✅ Quiz completed. Thanks!")