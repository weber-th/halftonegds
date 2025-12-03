import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
import logging
from src.halftoning import Halftoner
from src.gds_writer import GDSWriter
from src.vector_processing import parse_svg, trace_image, filter_polygons_by_area, boolean_operation, offset_polygons
import gdstk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Increase PIL Image limit to handle large files
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Halftone GDS Generator", layout="wide")

st.title("Halftone GDS Generator")
st.markdown("""
Convert images or vector files into GDSII patterns.
""")

# Sidebar settings
st.sidebar.header("Settings")

# --- Mode Selection ---
mode = st.sidebar.radio("Mode", ["Halftone", "Vector/Trace"])

# --- Wafer Settings ---
st.sidebar.subheader("Wafer Settings")
wafer_size_inch = st.sidebar.selectbox(
    "Wafer Size",
    options=[1, 2, 3, 4],
    index=0, # Default to 1 inch
    format_func=lambda x: f"{x} inch"
)

background_color = st.sidebar.slider(
    "Preview Background (grayscale)",
    min_value=0,
    max_value=255,
    value=255,
    help="Set the preview background color. Use a bright value (e.g. 255) for typical reflective wafers."
)


background_cleanup_tolerance = st.sidebar.slider(
    "Background Cleanup (tolerance)",
    min_value=0,
    max_value=50,
    value=10,
    help="Treat pixels within this range of the background color as background to remove faint JPG borders.",
)
pixel_size = st.sidebar.number_input(
    "Pixel Size (µm)", 
    min_value=0.1, 
    value=5.0, 
    step=0.1,
    help="Size of each pixel in the GDS file (or scaling factor for vectors)."
)

dpi = 25400 / pixel_size
st.sidebar.info(f"Effective DPI: {int(dpi)}")

edge_exclusion = st.sidebar.number_input(
    "Edge Exclusion (mm)",
    min_value=0.0,
    value=0.0,
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

# --- Mode Specific Settings ---
if mode == "Halftone":
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

elif mode == "Vector/Trace":
    st.sidebar.subheader("Vector/Trace Settings")
    
    # Trace Settings
    trace_threshold = st.sidebar.slider("Trace Threshold", 0, 255, 128, 1)
    epsilon_factor = st.sidebar.slider("Smoothness (Epsilon)", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
    fill_holes = st.sidebar.checkbox("Fill Holes (Trace)", value=False, help="If checked, ignores internal holes during tracing.")

    # Advanced Operations
    with st.sidebar.expander("Advanced Operations"):
        st.markdown("### Filtering")
        min_area = st.sidebar.number_input("Min Area (px²)", value=0.0, step=10.0)
        max_area = st.sidebar.number_input("Max Area (px²)", value=0.0, step=100.0, help="0 for no limit")
        
        st.markdown("### Boolean Ops")
        enable_union = st.sidebar.checkbox("Merge All (Union)", value=False, help="Merge overlapping polygons.")
        enable_invert = st.sidebar.checkbox("Invert (Negative)", value=False, help="Subtract patterns from the wafer.")
        
        st.markdown("### Offset")
        offset_val = st.sidebar.number_input("Offset (µm)", value=0.0, step=0.1, help="Positive to grow, negative to shrink.")


pattern_layer = st.sidebar.number_input("Pattern Layer", value=0, step=1)
outline_layer = st.sidebar.number_input("Outline Layer", value=2, step=1)


# File uploader
uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png", "bmp", "svg"])

def apply_gamma(image, gamma=1.0):
    if gamma == 1.0:
        return image
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def clean_background(image, background_color, tolerance):
    if tolerance <= 0:
        return image

    # Pixels that are already close to the intended background are forced to the background
    cutoff = np.clip(int(background_color) - tolerance, 0, 255)
    cleaned = image.copy()
    cleaned[cleaned >= cutoff] = background_color
    return cleaned

def process_image(image, wafer_size_inch, pixel_size, scale, rotation, offset_x_mm, offset_y_mm, edge_exclusion_mm, flat_orientation, background_color):
    logger.info("Processing image transformation...")
    # Wafer dimensions in microns
    wafer_diameter_um = wafer_size_inch * 25400
    
    # Target resolution (pixels)
    target_dim = int(wafer_diameter_um / pixel_size)
    
    # Create canvas with configurable background
    canvas = np.full((target_dim, target_dim), background_color, dtype=np.uint8)
    
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
    
    transformed = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_color,
    )
    
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
    # Keep the wafer region as-is but force the excluded area back to the chosen
    # background color so dark corners don't turn into features in the halftone/GDS.
    canvas[mask == 0] = background_color
        
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
if 'polygons' not in st.session_state:
    st.session_state['polygons'] = None

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    if mode == "Halftone":
        if file_ext == "svg":
            st.error("SVG files are not supported in Halftone mode. Please switch to Vector/Trace mode.")
        else:
            # Read image with alpha-awareness so transparent pixels default to bright
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            decoded = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            if decoded is None:
                st.error("Unable to decode the uploaded image.")
                st.stop()

            if len(decoded.shape) == 2:
                original_image = decoded
            elif decoded.shape[2] == 4:
                bgr = decoded[:, :, :3]
                alpha = decoded[:, :, 3] / 255.0
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                # Treat transparent pixels as bright by compositing on white
                original_image = (gray * alpha + 255 * (1 - alpha)).astype(np.uint8)
            else:
                original_image = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)

            # Apply Gamma Correction
            original_image = apply_gamma(original_image, gamma)

            # Clean near-background pixels so JPEG halos don't leave a border when placed on the wafer
            original_image = clean_background(original_image, background_color, background_cleanup_tolerance)

            if invert:
                original_image = 255 - original_image

            # Process Image (Scale, Rotate, Place on Wafer Canvas, Masking)
            processed_image = process_image(
                original_image, wafer_size_inch, pixel_size, scale, rotation,
                offset_x, offset_y, edge_exclusion, flat_orientation, background_color
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
                        halftone_output = halftoner.run(algorithm=algorithm)

                        # Dark features should correspond to written areas on a bright wafer,
                        # so invert the halftone result to make dark pixels the written pixels.
                        binary_image = 1 - halftone_output
                        st.session_state['binary_image'] = binary_image
                        logger.info("Halftoning complete.")

                # Display cached result if available
                if st.session_state['binary_image'] is not None:
                    display_image = np.full(st.session_state['binary_image'].shape, background_color, dtype=np.uint8)
                    display_image[st.session_state['binary_image'] == 1] = 0

                    st.image(display_image, caption=f"Result ({algorithm})", clamp=True)
                    
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
                        # Blur to simulate light diffraction/resist diffusion on dark features
                        simulated = cv2.GaussianBlur(display_image, (0, 0), sigmaX=1.0)
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

    elif mode == "Vector/Trace":
        col1, col2 = st.columns(2)
        
        polygons = []
        
        if file_ext == "svg":
            # SVG Processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp_svg:
                tmp_svg.write(uploaded_file.read())
                tmp_svg_path = tmp_svg.name
            
            try:
                polygons = parse_svg(tmp_svg_path, scale=scale) # Apply scale from sidebar
                st.session_state['polygons'] = polygons
            finally:
                os.unlink(tmp_svg_path)
                
        else:
            # Image Tracing
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            decoded = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            if decoded is None:
                st.error("Unable to decode the uploaded image.")
                st.stop()

            if len(decoded.shape) == 2:
                original_image = decoded
            elif decoded.shape[2] == 4:
                bgr = decoded[:, :, :3]
                alpha = decoded[:, :, 3] / 255.0
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                original_image = (gray * alpha + 255 * (1 - alpha)).astype(np.uint8)
            else:
                original_image = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)

            original_image = clean_background(original_image, background_color, background_cleanup_tolerance)

            # Apply transforms (Rotate/Scale/Offset) BEFORE tracing?
            # Or trace then transform?
            # Let's use the existing process_image to put it on the wafer canvas first
            # This handles scaling, rotation, offset, exclusion, etc.
            
            processed_image = process_image(
                original_image, wafer_size_inch, pixel_size, scale, rotation,
                offset_x, offset_y, edge_exclusion, flat_orientation, background_color
            )
            
            with col1:
                st.subheader("Preview (Wafer)")
                preview_img = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                preview_img = draw_wafer_outline(preview_img, wafer_size_inch, pixel_size, flat_orientation)
                st.image(preview_img, caption="Processed Image", clamp=True)
            
            # Trace
            polygons = trace_image(processed_image, threshold=trace_threshold, epsilon_factor=epsilon_factor)
            st.session_state['polygons'] = polygons

        with col2:
            st.subheader("Vector Preview")
            if st.session_state['polygons']:
                st.info(f"Found {len(st.session_state['polygons'])} polygons.")
                
                # Draw polygons on a blank canvas for preview
                wafer_diameter_um = wafer_size_inch * 25400
                target_dim = int(wafer_diameter_um / pixel_size)
                vector_preview = np.full((target_dim, target_dim, 3), background_color, dtype=np.uint8)
                
                # Draw polygons (Green)
                # Polygons are in pixel coordinates
                polys_int = [p.astype(np.int32) for p in st.session_state['polygons']]
                cv2.polylines(vector_preview, polys_int, True, (0, 255, 0), 1)
                
                st.image(vector_preview, caption="Vector Preview", clamp=True)
            else:
                st.warning("No polygons found.")

        st.markdown("---")
        
        if st.button("Generate GDS (Vector)"):
             if not st.session_state['polygons']:
                 st.error("No polygons to write!")
             else:
                with st.spinner("Generating GDS file..."):
                    try:
                        logger.info("Starting GDS generation (Vector)...")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".gds") as tmp_file:
                            tmp_filename = tmp_file.name
                        
                        # Initialize writer just to use the generate_from_polygons method
                        # Image is not needed for vector mode, pass dummy
                        dummy_img = np.zeros((10,10))
                        writer = GDSWriter(
                            dummy_img, 
                            pixel_size, 
                            wafer_size_inch,
                            pattern_layer=pattern_layer,
                            outline_layer=outline_layer
                        )
                        
                        lib = writer.generate_from_polygons(st.session_state['polygons'])
                        lib.write_gds(tmp_filename)
                        
                        logger.info(f"GDS saved to {tmp_filename}")
                        
                        with open(tmp_filename, "rb") as f:
                            gds_data = f.read()
                        
                        st.success(f"GDS generated successfully! ({len(gds_data)/1024:.1f} KB)")
                        
                        st.download_button(
                            label="Download GDS",
                            data=gds_data,
                            file_name="vector_output.gds",
                            mime="application/octet-stream"
                        )
                        
                        os.unlink(tmp_filename)
                        
                    except Exception as e:
                        logger.error(f"Error generating GDS: {e}")
                        st.error(f"Error generating GDS: {e}")

else:
    st.info("Please upload a file to get started.")
