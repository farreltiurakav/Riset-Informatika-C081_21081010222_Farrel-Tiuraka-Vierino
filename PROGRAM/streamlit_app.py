import streamlit as st
import cv2
from PIL import Image
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
SMILE_CASCADE = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect_faces(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dengan pengaturan parameter yang lebih hati-hati
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Sesuaikan nilai ini
        minNeighbors=5,   # Sesuaikan nilai ini
        minSize=(30, 30)  # Ukuran minimum wajah yang dapat terdeteksi
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 2, 4)

        # Deteksi mata dan senyuman di dalam area wajah yang terdeteksi
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    return img, faces

def main():
    """Face Detection App"""
    st.title("Face Detection App")
    st.text("Farrel Face Detection App")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Home':
        st.subheader("Face Detection")

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            image = Image.open(image_file)
            st.text("Original Image")
            st.image(image)

        task = ["Face Detection"]
        feature_choice = st.sidebar.selectbox("Task", task)
        if st.button("Process"):
            if feature_choice == 'Face Detection':
                result_img, result_face = detect_faces(image)
                st.success("Found {} faces".format(len(result_face)))
                st.image(result_img)

    elif choice == 'About':
        st.subheader("About Face Detection App")
        st.markdown("Farrel Face Detection App")
        st.text("Done")
        st.success("Success!!")

if __name__ == '__main__':
    main()
