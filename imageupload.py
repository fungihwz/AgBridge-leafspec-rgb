
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt




st.set_page_config(page_title="Leaf DGCI Classroom", layout="centered")
# Images at top
st.image("images/banner.jpg", use_container_width=True)
# Header
st.title("🌿 Our Garden Guardian!")
st.header("Uncovering secrets to healthy plant")

st.markdown("""
The goal of this app is to 
measure in a non-destructive way the Dark Green Color Index of leaf from its image.  
Refer to your teacher’s instructions on how you should take photos of leaves and where to record the results. 
Make sure to record the results of a leaf before uploading a new image. 
""")

# SECTION 1: Upload + DGCI computation
st.header("Upload Leaf Image")
uploaded = st.file_uploader("Upload a leaf image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded:
    # img = Image.open(uploaded).convert("RGB")
    # st.image(img, caption="Uploaded Image", use_container_width=True)
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # arr = np.array(img)  # img passed earlier from PIL
    # hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Step 1: Try to detect ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners_list, ids, _ = detector.detectMarkers(image)

    if ids is not None and len(ids) >= 4:
        st.info(f"ArUco markers detected: {ids.flatten().tolist()}")
        # --- ArUco ROI pipeline ---
        ids = ids.flatten()
        desired_order = [10, 20, 40, 30]
        centers = []


        for corner, marker_id in zip(corners_list, ids):
            M = cv2.moments(corner.reshape((-1, 1, 2)))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((marker_id, (cX, cY)))


        # Sort and extract ROI polygon
        sorted_centers = sorted(centers, key=lambda x: desired_order.index(x[0]))
        pts = np.array([c[1] for c in sorted_centers], dtype=np.int32).reshape((-1, 1, 2))


        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        roi = cv2.bitwise_and(image, image, mask=mask)
        working_img = roi

        # --- Debug visualization: show bounding quadrilateral ---
        debug_img = image.copy()
        cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 255), thickness=10)
        st.subheader("ArUco ROI Debug")
        st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    else:
        # st.warning("No ArUco markers detected – falling back to white background pipeline.")
        working_img = image
        
    # --- White background pipeline ---
    hsv = cv2.cvtColor(working_img, cv2.COLOR_BGR2HSV)
    lower_hue = np.array([20, 55, 42])
    upper_hue = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_hue, upper_hue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=2)


    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        st.warning("No leaf detected.")
        st.stop()

    leaf_contour = max(contours, key=cv2.contourArea)
    leaf_mask = np.zeros_like(mask_clean)
    cv2.drawContours(leaf_mask, [leaf_contour], -1, 255, thickness=-1)
        
    leaf_img = cv2.bitwise_and(working_img, working_img, mask=leaf_mask)
    
    # st.subheader("1. Original Image")
    # st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # DGCI pipeline
    hsv_leaf = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2HSV)
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_leaf)
    h = h.astype(np.float32) * 2
    s = s.astype(np.float32) / 255.0
    v = v.astype(np.float32) / 255.0
    dgci = np.clip(((h - 60) / 60 + (1 - s) + (1 - v)) / 3, 0, 1)
    masked_dgci = np.zeros_like(dgci)
    masked_dgci[leaf_mask > 0] = dgci[leaf_mask > 0]

    #heatmap = cv2.applyColorMap((dgci * 255).astype(np.uint8), cv2.COLORMAP_JET)
    final_heat = cv2.applyColorMap((masked_dgci * 255).astype(np.uint8), cv2.COLORMAP_JET)
    leaf_only = cv2.bitwise_and(final_heat, final_heat, mask=leaf_mask)

    contour_img = image.copy()
    cv2.drawContours(contour_img, [leaf_contour], -1, (255, 0, 0), 20)

    mean_dgci = masked_dgci[leaf_mask > 0].mean()
    # st.metric("Mean DGCI", f"{mean_dgci:.3f}")
    # Compute SPAD using the linear fit equation
    spad_estimate = 117.874 * mean_dgci - 11.8


    # Display both metrics side by side
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean DGCI", f"{mean_dgci:.3f}")
    with col2:
        st.metric("Estimated SPAD (for tomato only)", f"{spad_estimate:.2f}")


    cols = st.columns(2)
    with cols[0]:
        st.subheader("1. Original + Contour")
        st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with cols[1]:
        st.subheader("2. DGCI Heatmap")
        st.image(cv2.cvtColor(leaf_only, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Add heatmap legend below
        fig, ax = plt.subplots(figsize=(4, 0.5))
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=0, vmax=1)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
        cbar.set_label("DGCI Scale")
        st.pyplot(fig)






st.markdown("""
This app and its accompanying unit are designed by Sopheak Seng (sengs@purdue.edu) and Hanwei Zhou (zhou1013@purdue.edu) as part of the AgBridge project. 
The project is a part of IoT4Ag and funded by the National Science Foundation. 
""")

st.markdown("""
The accompanying unit can be found at IoT4Ag website 
""")