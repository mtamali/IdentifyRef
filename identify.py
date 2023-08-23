import streamlit as st
import cv2
import os
import tempfile
import zipfile

def identify_places(reference_image_path, image_folder_path):
    ref_img = cv2.imread(reference_image_path, 0)
    orb = cv2.ORB_create()
    ref_kp, ref_des = orb.detectAndCompute(ref_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matching_images = []

    for filename in os.listdir(image_folder_path):
        img_path = os.path.join(image_folder_path, filename)
        img = cv2.imread(img_path, 0)
        kp, des = orb.detectAndCompute(img, None)

        if des is not None:
            matches = bf.match(ref_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > 50:
                matching_images.append(img_path)

    return matching_images

def get_download_link(file_path, file_label='File'):
    href = f'<a href="{file_path}" target="_blank">Download {file_label}</a>'
    return href

st.title("Identify Place within Images")

reference_image_path = None

st.subheader("Upload Reference Image")
uploaded_file = st.file_uploader("Choose a reference image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    reference_image_path = tfile.name
    st.image(reference_image_path, caption="Reference Image", use_column_width=True)

st.subheader("Select Folder with Images")
image_folder_path = st.text_input("Enter the path to the folder containing the images:")

if reference_image_path and image_folder_path:
    if st.button('Begin Fetching'):
        matching_images = identify_places(reference_image_path, image_folder_path)

        num_matching_files = len(matching_images)
        st.write(f"Number of matching images: {num_matching_files}")

        if matching_images:
            zip_file_name = "matching.zip"
            zipf_path = os.path.join('/home/modeb/Pictures', zip_file_name) # Change 'static' to your public folder
            with zipfile.ZipFile(zipf_path, 'w') as zipf:
                for img_path in matching_images:
                    zipf.write(img_path, os.path.basename(img_path))

            download_link = get_download_link(f'/home/modeb/Pictures/{zip_file_name}', zip_file_name) # Change the path to the public URL
            st.markdown(download_link, unsafe_allow_html=True)
