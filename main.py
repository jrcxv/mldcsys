import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import tempfile
import matplotlib.pyplot as plt

# Set the theme colors
st.set_page_config(page_title="MLDC System", page_icon="🍃", layout="wide")
st.markdown(
    """
    <style>
    .css-18e3th9 {
        background-color: #123524;
    }
    .css-1d391kg {
        background-color: #3E7B27;
    }
    .css-1cpxqw2 {
        color: #EFE3C2;
    }
    .css-1cpxqw2 a {
        color: #85A947;
    }
    .css-1cpxqw2 a:hover {
        color: #EFE3C2;
    }
    .stButton>button {
        background-color: #3E7B27;
        color: #EFE3C2;
    }
    .stButton>button:hover {
        background-color: #85A947;
        color: #123524;
    }
    .card {
        background-color: #123524;
        padding: 20px;
        margin: 10px 0;
        border-radius: 10px;
        color: #EFE3C2;
        text-align: center;
    }
    .card a {
        color: #85A947;
        text-decoration: none;
    }
    .card a:hover {
        color: #EFE3C2;
    }
    .card img {
        width: 50px;
        height: 50px;
        margin-bottom: 10px;
    }
    .card-content {
        display: flex;
        flex-direction: column;
        align-items: center.
    }
    .card-content .label {
        font-size: 1.2em;
        font-weight: bold.
    }
    .card-content .description {
        font-size: 0.9em.
        font-family: Arial, sans-serif.
    }
    .css-1d391kg .stSelectbox [data-baseweb="select"] {
        background-color: #3E7B27.
        color: #EFE3C2.
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Disease descriptions and links
disease_info = {
    'Anthracnose': {
        'description': 'Anthracnose is a fungal disease that causes dark, sunken lesions on leaves, stems, flowers, and fruits.',
        'link': 'https://apps.lucidcentral.org/pppw_v10/text/web_full/entities/mango_anthracnose_009.htm'
    },
    'Bacterial Canker': {
        'description': 'Bacterial canker is a bacterial disease that causes sunken, water-soaked lesions on leaves, stems, and fruits.',
        'link': 'https://agriculture.vikaspedia.in/viewcontent/agriculture/crop-production/integrated-pest-managment/ipm-for-fruit-crops/ipm-strategies-for-mango/mango-diseases-and-symptoms?lgn=en'
    },
    'Cutting Weevil': {
        'description': 'Cutting weevil is an insect pest that cuts the leaves and stems of plants, causing significant damage.',
        'link': 'https://plantwiseplusknowledgebank.org/doi/full/10.1079/pwkb.species.18427'
    },
    'Die Back': {
        'description': 'Die back is a condition where the tips of the branches die back, often caused by fungal or bacterial infections.',
        'link': 'https://www.industry.mangoes.net.au/cmsb/media/pest_2_c-moore-mttd-presentation-final.pdf'
    },
    'Gall Midge': {
        'description': 'Gall midge is an insect pest that causes the formation of galls on leaves and stems, leading to deformation and damage.',
        'link': 'https://www.business.qld.gov.au/industries/farms-fishing-forestry/agriculture/biosecurity/plants/priority-pest-disease/mango-leaf-gall-midge#:~:text=Mango%20gall%20midge%20are%20tiny,%2C%20flowers%2C%20fruit%20and%20shoots.'
    },
    'Powdery Mildew': {
        'description': 'Powdery mildew is a fungal disease that causes a white, powdery coating on leaves, stems, and flowers.',
        'link': 'https://www.cropscience.bayer.eg/en-eg/pests/diseases/mango-powdery-mildew.html#:~:text=Oidium%20mangiferae%20Berthet%20(a%20fungus,be%20infected%20by%20the%20fungus).'
    },
    'Sooty Mould': {
        'description': 'Sooty mould is a fungal disease that causes a black, sooty coating on leaves, stems, and fruits, often associated with insect infestations.',
        'link': 'https://blogs.ifas.ufl.edu/stlucieco/2020/09/21/mango-tree-sooty-mold/#:~:text=Mango%20sooty%20mold%20(Meliola%20mangiferae,treehoppers%20and%20non%2Dparasitic%20fungi.'
    }
}

# Tensorflow Model Prediction
def model_prediction(test_image):
    # Load the model, specifying the format if necessary
    model = tf.keras.models.load_model('mdlsys_model.h5', compile=False)
    image = Image.open(test_image)
    image = image.resize((224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index, model, input_arr

# Generate heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {img_path}")
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path

def plot_gradcam(img_path, heatmap, title="Grad-CAM Heatmap"):
    img = Image.open(img_path)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize(img.size, Image.LANCZOS)
    superimposed_img = Image.blend(img, heatmap, alpha=0.4)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(superimposed_img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Useful Links"])

# Home Page
if app_mode == "Home":
    st.header("MANGO LEAF DISEASE CLASSIFICATION SYSTEM")
    image_path = "home_page.webp"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Mango Leaf Disease Classification System! 🌿🔍
    
    Identify diseases in mango leaves using advanced machine learning algorithms. Simply upload an image of a mango leaf, and the model will analyze it to detect potential diseases.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a mango leaf with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Mango Leaf Disease Classification System!
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset consists of images of mango leaves with various diseases. The dataset is divided into training, validation, and test sets. The images are categorized into different classes representing different diseases.
    
    #### Content
    - **Type of data:** 240x320 mango leaf images.
    - **Data format:** JPG.
    - **Number of images:** 4000 images. Of these, around 1800 are of distinct leaves, and the rest are prepared by zooming and rotating where deemed necessary.
    - **Diseases considered (classes):** Seven diseases, namely Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Healthy and Sooty Mould.
    - **Distribution of instances:** Each of the eight categories contains 500 images.
    - **How data are acquired:** Captured from mango trees through the mobile phone camera.
    - **Data source locations:** Four mango orchards of Bangladesh, namely Sher-e-Bangla Agricultural University orchard, Jahangir Nagar University orchard, Udaypur village mango orchard, and Itakhola village mango orchard.
    
    For more details, refer to:
    Ali, Sawkat; Ibrahim, Muhammad ; Ahmed, Sarder Iftekhar ; Nadim, Md. ; Mizanur, Mizanur Rahman; Shejunti, Maria Mehjabin ; Jabid, Taskeed (2022), “MangoLeafBD Dataset”, Mendeley Data, V1, doi: 10.17632/hxsnvwty3r.1
    """)

# Useful Links Page
elif app_mode == "Useful Links":
    st.header("Useful Links")
    st.markdown("""
    <div class="card">
        <a href="https://github.com/EnriqManComp/Mango-Leaf-Disease-Classification/" target="_blank">
            <div class="card-content">
                <img src="https://pngimg.com/uploads/github/github_PNG80.png" alt="GitHub" style="display: block; margin-left: auto; margin-right: auto;">
                <div class="label" style="font-weight: bold;">GitHub Repository</div>
                <div class="description">View the GitHub repository for the Jupyter notebook and more details.</div>
            </div>
        </a>
    </div>
    <div class="card">
        <a href="https://www.kaggle.com/code/enriquecompanioni/mango-leaf-disease-classification" target="_blank">
            <div class="card-content">
                <img src="https://www.kaggle.com/static/images/site-logo.png" alt="Kaggle" style="display: block; margin-left: auto; margin-right: auto;">
                <div class="label" style="font-weight: bold;">Kaggle Notebook</div>
                <div class="description">View the Kaggle notebook to learn more about how the model was trained and the output model.</div>
            </div>
        </a>
    </div>
    <div class="card">
        <a href="https://data.mendeley.com/datasets/hxsnvwty3r/1" target="_blank">
            <div class="card-content">
                <img src="https://cdn.freebiesupply.com/logos/large/2x/mendeley-2-logo-black-and-white.png" alt="Mendeley" style="display: block; margin-left: auto; margin-right: auto;">
                <div class="label" style="font-weight: bold;">Dataset</div>
                <div class="description">View the dataset used to train the model.</div>
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Mango Leaf Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_container_width=True)
    # Predict Button
    if st.button("Predict"):
        if test_image is not None:
            with st.spinner("Please Wait.."):
                st.write("Analyzing the image...")
                result_index, model, input_arr = model_prediction(test_image)
                # Define Class
                class_name = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
                if class_name[result_index] == 'Healthy':
                    st.success(f"The model predicts that the mango leaf is **{class_name[result_index]}**.")
                else:
                    st.success(f"The model predicts that the mango leaf has **{class_name[result_index]}**.")
                    
                    # Display disease information
                    disease = class_name[result_index]
                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-content">
                                <div class="label">{disease}</div>
                                <div class="description">{disease_info[disease]['description']}</div>
                                <a href="{disease_info[disease]['link']}" target="_blank">Read more</a>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Save the uploaded image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(test_image.getvalue())
                    temp_file_path = temp_file.name
                
                # Generate and display heatmap
                heatmap = make_gradcam_heatmap(input_arr, model, 'conv2d')
                plot_gradcam(temp_file_path, heatmap, title="Grad-CAM Heatmap")
                st.pyplot(plt)
                st.markdown("""
                ### Heatmap Interpretation
                The Grad-CAM heatmap highlights the regions of the image that are most important for the model's prediction. The areas with the highest intensity (red regions) indicate the parts of the image that contributed most to the model's decision. In the context of mango leaf disease classification, these regions likely correspond to the diseased areas of the leaf.
                """)