import gdstk
import numpy as np

class GDSWriter:
    def __init__(self, binary_image, pixel_size_um, wafer_size_inch=4, shape="rectangle", pattern_layer=0, outline_layer=2):
        """
        Initialize with a binary image (numpy array) and pixel size in microns.
        binary_image: 2D numpy array where 1 represents a pixel to be drawn.
        pixel_size_um: Size of each pixel in microns.
        wafer_size_inch: Size of the wafer in inches.
        shape: "rectangle" or "circle".
        pattern_layer: GDS layer for the halftone pattern.
        outline_layer: GDS layer for the wafer outline.
        """
        self.image = binary_image
        self.pixel_size = pixel_size_um
        self.wafer_size_inch = wafer_size_inch
        self.shape = shape
        self.pattern_layer = pattern_layer
        self.outline_layer = outline_layer
        self.height, self.width = self.image.shape

    def generate_library(self, lib_name="HalftoneLib", cell_name="Main"):
        """
        Generate a GDS library containing the halftone pattern.
        Returns the gdstk.Library object.
        """
        lib = gdstk.Library(lib_name)
        cell = lib.new_cell(cell_name)
        
        # 1. Draw Wafer Outline
        wafer_diameter_um = self.wafer_size_inch * 25400
        wafer_radius_um = wafer_diameter_um / 2
        
        # Circle centered at (0,0) - Draw as a ring (outline)
        # inner_radius = radius - 100um (just a thin line)
        wafer_outline = gdstk.ellipse((0, 0), wafer_radius_um, inner_radius=wafer_radius_um-100, layer=self.outline_layer, tolerance=1.0)
        cell.add(wafer_outline)
        
        # 2. Draw Halftone Pattern
        # The image is already sized to the wafer (from app.py logic), 
        # so we just need to center it at (0,0).
        # Image (0,0) is top-left.
        # Total width/height in microns
        total_width_um = self.width * self.pixel_size
        total_height_um = self.height * self.pixel_size
        
        # Offset to move top-left (0,0) to (-w/2, h/2)
        start_x = -total_width_um / 2
        start_y = total_height_um / 2
        
        if self.shape == "rectangle":
            # Optimization: Run-Length Encoding for rectangles
            for r in range(self.height):
                y_top = start_y - r * self.pixel_size
                y_bottom = start_y - (r + 1) * self.pixel_size
                
                row_data = self.image[r]
                
                # Find runs of 1s
                padded = np.concatenate(([0], row_data, [0]))
                diff = np.diff(padded)
                
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                for start, end in zip(starts, ends):
                    x_left = start_x + start * self.pixel_size
                    x_right = start_x + end * self.pixel_size
                    
                    rect = gdstk.rectangle((x_left, y_bottom), (x_right, y_top), layer=self.pattern_layer)
                    cell.add(rect)
                    
        elif self.shape == "circle":
            # No RLE possible for circles, must draw individually
            # This will be slower and generate larger files
            radius = self.pixel_size / 2
            
            # Get indices of all 1s
            ys, xs = np.where(self.image == 1)
            
            for y, x in zip(ys, xs):
                center_x = start_x + (x + 0.5) * self.pixel_size
                center_y = start_y - (y + 0.5) * self.pixel_size
                
                circle = gdstk.ellipse((center_x, center_y), radius, layer=self.pattern_layer, tolerance=0.1)
                cell.add(circle)
                
        return lib

    def save(self, filename, lib_name="HalftoneLib", cell_name="Main"):
        """
        Generate and save the GDS file.
        """
        lib = self.generate_library(lib_name, cell_name)
        lib.write_gds(filename)
        return filename
