from streamlit import st
from yolov8.model import load_model, perform_inference

def main():
    st.title("YOLOv8 Streamlit App")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = uploaded_file.read()
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        model = load_model()
        results = perform_inference(model, image)
        
        st.subheader("Inference Results")
        st.write(results)

if __name__ == "__main__":
    main()