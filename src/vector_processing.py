import numpy as np
import cv2
from svgpathtools import svg2paths

def parse_svg(svg_file_path, scale=1.0):
    """
    Parse an SVG file and extract paths as polygons.
    Returns a list of polygons (numpy arrays of shape (N, 2)).
    """
    paths, attributes = svg2paths(svg_file_path)
    polygons = []
    
    for path in paths:
        # Sample the path to create a polygon
        # Number of samples depends on path length, but let's use a fixed resolution for now
        # or adaptive sampling. svgpathtools allows sampling.
        
        # Calculate length to determine number of samples
        length = path.length()
        if length == 0:
            continue
            
        # Sample points. Resolution: 1 point per unit length (approx)
        # Adjust this based on your needs.
        num_samples = max(int(length), 10) 
        
        points = []
        for i in range(num_samples + 1):
            t = i / num_samples
            pt = path.point(t)
            points.append([pt.real, pt.imag])
            
        poly = np.array(points) * scale
        
        # Invert Y axis because SVG usually has Y down, GDS has Y up (or we handle it in GDSWriter)
        # But usually we want to keep image coordinates consistent.
        # Let's keep it as is and handle coordinate transform in GDSWriter or App.
        
        polygons.append(poly)
        
    return polygons

def trace_image(image, threshold=128, epsilon_factor=0.001, fill_holes=False):
    """
    Trace a bitmap image to find contours (polygons).
    image: Grayscale numpy array.
    threshold: Threshold value for binarization.
    epsilon_factor: Approximation accuracy (lower = more detailed).
    fill_holes: If True, only outer contours are retrieved (filling holes).
    """
    # 1. Binarize
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # 2. Find Contours
    # RETR_LIST retrieves all contours. RETR_EXTERNAL only outer.
    mode = cv2.RETR_EXTERNAL if fill_holes else cv2.RETR_LIST
    contours, hierarchy = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # 3. Approximate Contour to reduce points
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to simple (N, 2) array
        poly = approx.reshape(-1, 2)
        
        # Filter small noise
        if len(poly) >= 3:
            polygons.append(poly)
            
    return polygons

def filter_polygons_by_area(polygons, min_area=0, max_area=None):
    """
    Filter polygons by area.
    polygons: List of numpy arrays (N, 2).
    min_area: Minimum area to keep.
    max_area: Maximum area to keep (None for no limit).
    """
    filtered = []
    for poly in polygons:
        # Use cv2.contourArea for area calculation (works for simple polygons)
        # Ensure poly is float32 for cv2
        area = cv2.contourArea(poly.astype(np.float32))
        
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
            
        filtered.append(poly)
    return filtered

import gdstk

def boolean_operation(polygons, operation="union", operand_polygons=None):
    """
    Perform boolean operations on polygons using gdstk.
    polygons: List of numpy arrays (N, 2).
    operation: "union", "intersection", "difference", "xor".
    operand_polygons: Second set of polygons for binary operations.
    """
    # Convert numpy polygons to gdstk.Polygon
    gds_polys_1 = [gdstk.Polygon(p) for p in polygons]
    
    if operand_polygons:
        gds_polys_2 = [gdstk.Polygon(p) for p in operand_polygons]
    else:
        gds_polys_2 = []

    if operation == "union":
        # Union of all polygons in the list
        result = gdstk.boolean(gds_polys_1, gds_polys_2, "or")
    elif operation == "intersection":
        result = gdstk.boolean(gds_polys_1, gds_polys_2, "and")
    elif operation == "difference":
        result = gdstk.boolean(gds_polys_1, gds_polys_2, "not")
    elif operation == "xor":
        result = gdstk.boolean(gds_polys_1, gds_polys_2, "xor")
    else:
        return polygons

    # Convert back to numpy arrays
    return [p.points for p in result]

def offset_polygons(polygons, distance, tolerance=0.01):
    """
    Offset polygons (grow/shrink).
    distance: Offset distance (positive to grow, negative to shrink).
    """
    gds_polys = [gdstk.Polygon(p) for p in polygons]
    result = gdstk.offset(gds_polys, distance, tolerance=tolerance)
    return [p.points for p in result]
