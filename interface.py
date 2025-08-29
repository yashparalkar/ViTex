import streamlit as st
import torch
import pickle
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import time

# Import your model classes - make sure these files are in the same directory
from model import EncoderCNN, DecoderRNN
from build_vocab import Vocabulary

# Page configuration
st.set_page_config(
    page_title="Image Captioning",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 3px dashed rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.1);
    }
    
    .upload-section:hover {
        border-color: rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
        transform: translateY(-2px);
    }
    
    .caption-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .caption-text {
        font-size: 1.6rem;
        font-style: italic;
        color: #2d3748;
        text-align: center;
        line-height: 1.8;
        font-weight: 500;
        margin: 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .model-info {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid #90cdf4;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 10px rgba(66, 153, 225, 0.1);
    }
    
    .model-info strong {
        color: #2b6cb0;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 3rem;
        font-size: 1.2rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.4s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8 0%, #6b5b95 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Enhanced file uploader styling */
    .stFileUploader > div > div {
        border-radius: 15px;
        border: 2px dashed rgba(102, 126, 234, 0.4);
        background: rgba(102, 126, 234, 0.02);
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration - Update these paths to match your setup
CONFIG = {
    'encoder_path': 'models/encoder-5-3000.ckpt',
    'decoder_path': 'models/decoder-5-3000.ckpt', 
    'vocab_path': 'data/vocab.pkl',
    'embed_size': 256,
    'hidden_size': 512,
    'num_layers': 1
}

@st.cache_resource
def load_model():
    """Load your trained ResNet+LSTM model"""
    try:
        # Device configuration (matching your setup)
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            device_name = "Apple Silicon (MPS)"
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = "NVIDIA GPU (CUDA)"
        else:
            device = torch.device('cpu')
            device_name = "CPU"
        
        # Load vocabulary
        if not os.path.exists(CONFIG['vocab_path']):
            st.error(f"❌ Vocabulary file not found: {CONFIG['vocab_path']}")
            return None, None, None, None
            
        with open(CONFIG['vocab_path'], 'rb') as f:
            vocab = pickle.load(f)
        
        # Build models
        encoder = EncoderCNN(CONFIG['embed_size']).eval()
        decoder = DecoderRNN(CONFIG['embed_size'], CONFIG['hidden_size'], 
                           len(vocab), CONFIG['num_layers'])
        
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        # Load trained parameters
        if not os.path.exists(CONFIG['encoder_path']):
            st.error(f"❌ Encoder model not found: {CONFIG['encoder_path']}")
            return None, None, None, None
            
        if not os.path.exists(CONFIG['decoder_path']):
            st.error(f"❌ Decoder model not found: {CONFIG['decoder_path']}")
            return None, None, None, None
        
        encoder.load_state_dict(torch.load(CONFIG['encoder_path'], map_location=device))
        decoder.load_state_dict(torch.load(CONFIG['decoder_path'], map_location=device))
        
        st.success(f"✅ Model loaded successfully on {device_name}!")
        
        return encoder, decoder, vocab, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None

def preprocess_image(image):
    """Preprocess image exactly like in your original code"""
    # Convert to RGB and resize (matching your load_image function)
    image = image.convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    # Apply same transforms as your code
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def generate_caption(encoder, decoder, vocab, image_tensor, device):
    """Generate caption using your trained model"""
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            
            # Generate caption (matching your inference code)
            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)
            
            # Convert word_ids to words (matching your code)
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            
            # Join words and clean up
            sentence = ' '.join(sampled_caption)
            
            # Remove start/end tokens for cleaner output
            sentence = sentence.replace('<start>', '').replace('<end>', '').strip()
            
            return sentence
            
    except Exception as e:
        raise Exception(f"Error generating caption: {str(e)}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ ViTex Image Captioning</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload an image and let your trained AI model generate a descriptive caption</p>', unsafe_allow_html=True)
    
    # Load model
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if not st.session_state.model_loaded:
        with st.spinner("Loading your trained model..."):
            encoder, decoder, vocab, device = load_model()
            if encoder is not None:
                st.session_state.encoder = encoder
                st.session_state.decoder = decoder
                st.session_state.vocab = vocab
                st.session_state.device = device
                st.session_state.model_loaded = True
            else:
                st.stop()
    
    # Model information
    # if st.session_state.model_loaded:
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         st.markdown('<div class="model-info">', unsafe_allow_html=True)
    #         st.markdown(f"**Embed Size:** {CONFIG['embed_size']}")
    #         st.markdown(f"**Hidden Size:** {CONFIG['hidden_size']}")
    #         st.markdown('</div>', unsafe_allow_html=True)
    #     with col2:
    #         st.markdown('<div class="model-info">', unsafe_allow_html=True)
    #         st.markdown(f"**LSTM Layers:** {CONFIG['num_layers']}")
    #         st.markdown(f"**Vocab Size:** {len(st.session_state.vocab)}")
    #         st.markdown('</div>', unsafe_allow_html=True)
    #     with col3:
    #         st.markdown('<div class="model-info">', unsafe_allow_html=True)
    #         device_type = str(st.session_state.device).upper()
    #         st.markdown(f"**Device:** {device_type}")
    #         st.markdown(f"**Status:** ✅ Ready")
    #         st.markdown('</div>', unsafe_allow_html=True)
    
    # File uploader
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Supported formats: PNG, JPG, JPEG, GIF, BMP"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Create columns for image and info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display uploaded image with better styling
            image = Image.open(uploaded_file)
            st.image(image, caption="", use_column_width=True, 
                    output_format="auto")
        
        with col2:
            # Image information with enhanced styling
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**📸 Image Details**")
            st.markdown(f"**Size:** {image.size[0]} × {image.size[1]} px")
            st.markdown(f"**Format:** {image.format}")
            file_size = len(uploaded_file.getvalue())
            if file_size > 1024*1024:
                size_str = f"{file_size/(1024*1024):.1f} MB"
            elif file_size > 1024:
                size_str = f"{file_size/1024:.1f} KB"
            else:
                size_str = f"{file_size} bytes"
            st.markdown(f"**Size:** {size_str}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Generate caption button
        if st.button("🚀 Generate Caption", type="primary"):
            if not st.session_state.model_loaded:
                st.error("❌ Model not loaded properly. Please refresh the page.")
            else:
                try:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Preprocess image
                    status_text.text("🔄 Preprocessing image...")
                    progress_bar.progress(25)
                    image_tensor = preprocess_image(image)
                    
                    # Generate caption
                    status_text.text("🤖 Generating caption with ResNet + LSTM...")
                    progress_bar.progress(50)
                    
                    caption = generate_caption(
                        st.session_state.encoder,
                        st.session_state.decoder,
                        st.session_state.vocab,
                        image_tensor,
                        st.session_state.device
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Caption generated successfully!")
                    
                    # Display result
                    st.markdown('<div class="caption-box">', unsafe_allow_html=True)
                    st.markdown(f'<p class="caption-text">"{caption}"</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Clear progress indicators
                    time.sleep(1.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"❌ Error generating caption: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 0.9rem;'>Powered by ResNet + LSTM Architecture</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()