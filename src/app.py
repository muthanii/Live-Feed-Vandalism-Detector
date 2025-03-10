from streamlit import st
from yolov8.model import load_model, perform_inference


def main():
    st.title("YOLOv8 Object Detection with Streamlit")

    st.sidebar.header("Settings")
    model_option = st.sidebar.selectbox("Select Model", ["YOLOv8"])

    if st.sidebar.button("Load Model"):
        model = load_model(model_option)
        st.success("Model loaded successfully!")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        results = perform_inference(uploaded_file, model)
        st.write("Results:")
        st.json(results)


if __name__ == "__main__":
    main()
