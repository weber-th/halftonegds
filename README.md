# Halftone GDS Generator

A Streamlit application for converting images into halftone patterns and exporting them as GDSII files for lithography.

## Features

*   **Halftoning Algorithms**: Floyd-Steinberg, Atkinson, Jarvis-Judice-Ninke, Stucki, Burkes, Sierra Lite, and Bayer Ordered Dithering.
*   **Wafer Layout**:
    *   Support for 1", 2", 3", and 4" wafers.
    *   Adjustable **Edge Exclusion** zone.
    *   Configurable **Flat Orientation** (Bottom, Top, Left, Right).
*   **Image Processing**:
    *   **Gamma Correction** for contrast adjustment.
    *   **Invert** option for negative/positive resist.
    *   **Alignment**: Scale, Rotate, and Offset (X/Y) the image.
*   **GDSII Generation**:
    *   **Optimized**: Uses `gdstk` and Run-Length Encoding (RLE) for fast generation of rectangular pixels.
    *   **Shape Selection**: Choose between **Rectangle** (fast) or **Circle** pixels.
    *   **Layer Control**: Custom layers for the pattern and wafer outline.
*   **Analysis**:
    *   **Fill Factor**: Real-time calculation of pattern density.
    *   **Litho Simulation**: Preview of simulated resist development.
*   **Vector & Trace Mode**:
    *   **SVG Support**: Import SVG files directly.
    *   **Image Tracing**: Trace bitmaps to vectors with adjustable threshold and smoothness.
    *   **Advanced Operations**:
        *   **Filtering**: Remove polygons by area (Min/Max).
        *   **Boolean Ops**: Union (Merge), Difference (Invert/Negative).
        *   **Offset**: Grow or shrink polygons.
*   **Performance**: Accelerated halftoning using `numba` JIT compilation.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd halftonegds
    ```

2.  Install dependencies:

    **Using pip:**
    ```bash
    pip install -r requirements.txt
    ```

    **Using Mamba/Conda:**
    ```bash
    mamba env create -f environment.yml
    mamba activate halftonegds
    ```

## Usage

1.  Run the application:
    ```bash
    streamlit run app.py
    ```

2.  Open your browser to the provided URL (usually `http://localhost:8501`).

3.  **Workflow**:
    *   **Upload**: Drag and drop your image.
    *   **Configure Wafer**: Set size, pixel size (DPI), and exclusion zone.
    *   **Align**: Position your image on the wafer.
    *   **Process**: Adjust gamma and choose a halftoning algorithm.
    *   **Halftone**: Click "Run Halftoning" to generate the pattern.
    *   **Analyze**: Check the Fill Factor and Litho Simulation.
    *   **Export**: Click "Generate GDS" to download the file.

4.  **Vector/Trace Workflow**:
    *   **Select Mode**: Choose "Vector/Trace" in the sidebar.
    *   **Upload**: Upload an SVG or Image.
    *   **Trace (if Image)**: Adjust threshold and smoothness. Option to **Fill Holes**.
    *   **Advanced Ops**: Open the expander to Filter, Merge, Invert, or Offset polygons.
    *   **Export**: Click "Generate GDS (Vector)".

## Project Structure

*   `app.py`: Main Streamlit application.
*   `src/halftoning.py`: Halftoning algorithms (optimized with Numba).
*   `src/vector_processing.py`: SVG parsing, image tracing, and vector operations.
*   `src/gds_writer.py`: GDSII generation logic using `gdstk`.
*   `verify_logic.py`: Script for verifying core logic without the UI.
*   `requirements.txt` / `environment.yml`: Dependency definitions.

## License

[MIT License](LICENSE)
