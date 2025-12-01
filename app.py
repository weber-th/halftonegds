import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
import logging
from src.halftoning import Halftoner
from src.gds_writer import GDSWriter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Increase PIL Image limit to handle large files
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Halftone GDS Generator", layout="wide")

st.title("Halftone GDS Generator")
st.markdown("""
Upload an image to convert it into a halftone pattern and export it as a GDSII file.
""")

# Sidebar settings
st.sidebar.header("Settings")

# --- Wafer Settings ---
st.sidebar.subheader("Wafer Settings")
wafer_size_inch = st.sidebar.selectbox(
    "Wafer Size",
    options=[1, 2, 3, 4],
    index=3, # Default to 4 inch
    format_func=lambda x: f"{x} inch"
)

pixel_size = st.sidebar.number_input(
    "Pixel Size (µm)", 
    min_value=0.1, 
    value=10.0, 
    step=0.1,
    help="Size of each pixel in the GDS file."
)

dpi = 25400 / pixel_size
st.sidebar.info(f"Effective DPI: {int(dpi)}")

edge_exclusion = st.sidebar.number_input(
    "Edge Exclusion (mm)",
    min_value=0.0,
    value=2.0,
    step=0.1,
    help="Width of the safe zone around the wafer edge where no pattern is written."
)

flat_orientation = st.sidebar.selectbox(
    "Flat Orientation",
    options=["Bottom", "Top", "Left", "Right"],
    index=0
)

# --- Alignment Settings ---
st.sidebar.subheader("Alignment")
scale = st.sidebar.slider("Scale", 0.1, 5.0, 1.0, 0.01)
rotation = st.sidebar.slider("Rotation (degrees)", -180.0, 180.0, 0.0, 0.1)
offset_x = st.sidebar.number_input("Offset X (mm)", value=0.0, step=0.1)
offset_y = st.sidebar.number_input("Offset Y (mm)", value=0.0, step=0.1)

# --- Image Processing ---
st.sidebar.subheader("Image Processing")
gamma = st.sidebar.slider(
    "Gamma Correction",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Adjust image contrast. < 1.0 makes darks brighter, > 1.0 makes darks darker."
)

invert = st.sidebar.checkbox("Invert Image", value=False)

# --- Halftoning Settings ---
st.sidebar.subheader("Halftoning")
algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["floyd-steinberg", "atkinson", "jarvis-judice-ninke", "stucki", "burkes", "sierra-lite", "bayer"],
    index=0
)

# --- GDS Settings ---
st.sidebar.subheader("GDS Settings")
gds_shape = st.sidebar.selectbox(
    "Pixel Shape",
    options=["rectangle", "circle"],
    index=0,
    help="Shape of the individual pixels in the GDS file."
)

pattern_layer = st.sidebar.number_input("Pattern Layer", value=0, step=1)
outline_layer = st.sidebar.number_input("Outline Layer", value=2, step=1)


# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

def apply_gamma(image, gamma=1.0):
    if gamma == 1.0:
        return image
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_image(image, wafer_size_inch, pixel_size, scale, rotation, offset_x_mm, offset_y_mm, edge_exclusion_mm, flat_orientation):
    logger.info("Processing image transformation...")
    # Wafer dimensions in microns
    wafer_diameter_um = wafer_size_inch * 25400
    
    # Target resolution (pixels)
    target_dim = int(wafer_diameter_um / pixel_size)
    
    # Create canvas (black background)
    canvas = np.zeros((target_dim, target_dim), dtype=np.uint8)
    
    # Image transformation
    h, w = image.shape
    center = (w // 2, h // 2)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, rotation, scale)
    
    # Apply rotation and scaling
    # Calculate new bounding box to avoid clipping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    transformed = cv2.warpAffine(image, M, (new_w, new_h))
    
    # Place on canvas (centering and offset)
    # Offset in pixels
    off_x_px = int(offset_x_mm * 1000 / pixel_size)
    off_y_px = int(offset_y_mm * 1000 / pixel_size) 
    
    # Canvas center
    cx, cy = target_dim // 2, target_dim // 2
    
    tx = cx + off_x_px - new_w // 2
    ty = cy - off_y_px - new_h // 2
    
    # Blitting with bounds checking
    # Source region
    sx1, sy1 = 0, 0
    sx2, sy2 = new_w, new_h
    
    # Destination region
    dx1, dy1 = tx, ty
    dx2, dy2 = tx + new_w, ty + new_h
    
    # Clip to canvas
    if dx1 < 0:
        sx1 -= dx1
        dx1 = 0
    if dy1 < 0:
        sy1 -= dy1
        dy1 = 0
    if dx2 > target_dim:
        sx2 -= (dx2 - target_dim)
        dx2 = target_dim
    if dy2 > target_dim:
        sy2 -= (dy2 - target_dim)
        dy2 = target_dim
        
    if sx1 < sx2 and sy1 < sy2:
        canvas[dy1:dy2, dx1:dx2] = transformed[sy1:sy2, sx1:sx2]
    
    # --- Apply Wafer Mask (Edge Exclusion & Flat) ---
    # Create a mask where 255 is valid area, 0 is excluded
    mask = np.zeros((target_dim, target_dim), dtype=np.uint8)
    
    # 1. Circle with Edge Exclusion
    radius_px = target_dim // 2
    exclusion_px = int(edge_exclusion_mm * 1000 / pixel_size)
    valid_radius = radius_px - exclusion_px
    
    if valid_radius > 0:
        cv2.circle(mask, (cx, cy), valid_radius, 255, -1)
    
    # 2. Flat
    # Standard flat length is roughly 30-50mm depending on wafer size, 
    # but usually defined by a chord. 
    # Let's approximate a standard flat cut.
    # A simple way is to cut off a segment of the circle.
    # Let's say the flat is at 95% of the radius (just a visual approx for now, or we could be precise).
    # Standard primary flat is usually perpendicular to radius.
    
    flat_cut_ratio = 0.95 # Cut off outer 5%
    cut_dist = int(radius_px * flat_cut_ratio)
    
    if flat_orientation == "Bottom":
        mask[cy + cut_dist:, :] = 0
    elif flat_orientation == "Top":
        mask[:cy - cut_dist, :] = 0
    elif flat_orientation == "Left":
        mask[:, :cx - cut_dist] = 0
    elif flat_orientation == "Right":
        mask[:, cx + cut_dist:] = 0
        
    # Apply mask to canvas
    canvas = cv2.bitwise_and(canvas, canvas, mask=mask)
        
    return canvas

def draw_wafer_outline(image_rgb, wafer_size_inch, pixel_size, flat_orientation):
    # Draw wafer circle and flat for visualization
    h, w, _ = image_rgb.shape
    center = (w // 2, h // 2)
    radius = w // 2
    
    # Draw circle (green)
    cv2.circle(image_rgb, center, radius, (0, 255, 0), 2)
    
    # Draw flat
    flat_cut_ratio = 0.95
    cut_dist = int(radius * flat_cut_ratio)
    
    # Calculate chord points
    # x^2 + y^2 = r^2
    # if y = cut_dist, x = sqrt(r^2 - cut_dist^2)
    chord_half_len = int(np.sqrt(radius**2 - cut_dist**2))
    
    pt1, pt2 = None, None
    
    if flat_orientation == "Bottom":
        y = center[1] + cut_dist
        pt1 = (center[0] - chord_half_len, y)
        pt2 = (center[0] + chord_half_len, y)
    elif flat_orientation == "Top":
        y = center[1] - cut_dist
        pt1 = (center[0] - chord_half_len, y)
        pt2 = (center[0] + chord_half_len, y)
    elif flat_orientation == "Left":
        x = center[0] - cut_dist
        pt1 = (x, center[1] - chord_half_len)
        pt2 = (x, center[1] + chord_half_len)
    elif flat_orientation == "Right":
        x = center[0] + cut_dist
        pt1 = (x, center[1] - chord_half_len)
        pt2 = (x, center[1] + chord_half_len)
        
    if pt1 and pt2:
        cv2.line(image_rgb, pt1, pt2, (0, 255, 0), 2)
    
    return image_rgb

# Initialize session state for cached result
if 'binary_image' not in st.session_state:
    st.session_state['binary_image'] = None

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gamma Correction
    original_image = apply_gamma(original_image, gamma)
    
    if invert:
        original_image = 255 - original_image

    # Process Image (Scale, Rotate, Place on Wafer Canvas, Masking)
    processed_image = process_image(
        original_image, wafer_size_inch, pixel_size, scale, rotation, 
        offset_x, offset_y, edge_exclusion, flat_orientation
    )

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Preview (Wafer)")
        # Convert to RGB for visualization (to draw colored overlay)
        preview_img = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        preview_img = draw_wafer_outline(preview_img, wafer_size_inch, pixel_size, flat_orientation)
        st.image(preview_img, caption=f"Wafer: {wafer_size_inch}\"", clamp=True)

    with col2:
        st.subheader("Halftone Result")
        
        # Manual Trigger for Halftoning
        if st.button("Run Halftoning"):
            with st.spinner("Halftoning..."):
                logger.info(f"Starting halftoning with algorithm: {algorithm}")
                halftoner = Halftoner(processed_image)
                binary_image = halftoner.run(algorithm=algorithm)
                st.session_state['binary_image'] = binary_image
                logger.info("Halftoning complete.")
        
        # Display cached result if available
        if st.session_state['binary_image'] is not None:
            st.image(st.session_state['binary_image'] * 255, caption=f"Result ({algorithm})", clamp=True)
            
            # --- Analysis ---
            st.markdown("### Analysis")
            
            # Fill Factor
            total_pixels = st.session_state['binary_image'].size
            filled_pixels = np.count_nonzero(st.session_state['binary_image'])
            fill_factor = (filled_pixels / total_pixels) * 100
            st.metric("Fill Factor", f"{fill_factor:.2f}%")
            
            # Litho Simulation
            if st.checkbox("Show Lithography Simulation"):
                st.caption("Simulated resist development (Gaussian Blur)")
                # Blur to simulate light diffraction/resist diffusion
                # Sigma depends on resolution, just a visual guess here
                simulated = cv2.GaussianBlur(st.session_state['binary_image'] * 255, (0, 0), sigmaX=1.0)
                st.image(simulated, caption="Litho Simulation", clamp=True)
            
        else:
            st.info("Click 'Run Halftoning' to generate the pattern.")

    st.markdown("---")
    
    # GDS Generation
    if st.button("Generate GDS"):
        if st.session_state['binary_image'] is None:
            st.error("Please run halftoning first!")
        else:
            with st.spinner("Generating GDS file..."):
                try:
                    logger.info("Starting GDS generation...")
                    # Create a temporary file for the GDS
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".gds") as tmp_file:
                        tmp_filename = tmp_file.name
                    
                    writer = GDSWriter(
                        st.session_state['binary_image'], 
                        pixel_size, 
                        wafer_size_inch,
                        shape=gds_shape,
                        pattern_layer=pattern_layer,
                        outline_layer=outline_layer
                    )
                    writer.save(tmp_filename)
                    logger.info(f"GDS saved to {tmp_filename}")
                    
                    # Read the file back to allow download
                    with open(tmp_filename, "rb") as f:
                        gds_data = f.read()
                    
                    st.success(f"GDS generated successfully! ({len(gds_data)/1024:.1f} KB)")
                    
                    st.download_button(
                        label="Download GDS",
                        data=gds_data,
                        file_name="halftone_output.gds",
                        mime="application/octet-stream"
                    )
                    
                    # Cleanup
                    os.unlink(tmp_filename)
                    logger.info("Temporary file cleaned up.")
                    
                except Exception as e:
                    logger.error(f"Error generating GDS: {e}")
                    st.error(f"Error generating GDS: {e}")

else:
    st.info("Please upload an image to get started.")
