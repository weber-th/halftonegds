import numpy as np
import cv2
from numba import jit

@jit(nopython=True)
def _floyd_steinberg_jit(img):
    height, width = img.shape
    out = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            new_pixel = 1 if old_pixel > 0.5 else 0
            out[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            if x + 1 < width:
                img[y, x + 1] += quant_error * 7 / 16
            if x - 1 >= 0 and y + 1 < height:
                img[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < height:
                img[y + 1, x] += quant_error * 5 / 16
            if x + 1 < width and y + 1 < height:
                img[y + 1, x + 1] += quant_error * 1 / 16
    return out

@jit(nopython=True)
def _atkinson_jit(img):
    height, width = img.shape
    out = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            new_pixel = 1 if old_pixel > 0.5 else 0
            out[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            # Atkinson diffusion pattern (1/8)
            # Neighbors: (x+1,y), (x+2,y), (x-1,y+1), (x,y+1), (x+1,y+1), (x,y+2)
            
            if x + 1 < width:
                img[y, x + 1] += quant_error / 8
            if x + 2 < width:
                img[y, x + 2] += quant_error / 8
            if x - 1 >= 0 and y + 1 < height:
                img[y + 1, x - 1] += quant_error / 8
            if y + 1 < height:
                img[y + 1, x] += quant_error / 8
            if x + 1 < width and y + 1 < height:
                img[y + 1, x + 1] += quant_error / 8
            if y + 2 < height:
                img[y + 2, x] += quant_error / 8
                
    return out

@jit(nopython=True)
def _jarvis_judice_ninke_jit(img):
    height, width = img.shape
    out = np.zeros((height, width), dtype=np.uint8)
    
    # Diffusion matrix (relative coordinates, weight/48)
    # (1, 0, 7), (2, 0, 5),
    # (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
    # (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)
    
    # Numba doesn't like list of tuples iteration efficiently sometimes, unrolling or simple arrays is better.
    # But let's try explicit checks for clarity and speed.
    
    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            new_pixel = 1 if old_pixel > 0.5 else 0
            out[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            # Row 0 (y)
            if x + 1 < width: img[y, x + 1] += quant_error * 7 / 48
            if x + 2 < width: img[y, x + 2] += quant_error * 5 / 48
            
            # Row 1 (y+1)
            if y + 1 < height:
                if x - 2 >= 0: img[y + 1, x - 2] += quant_error * 3 / 48
                if x - 1 >= 0: img[y + 1, x - 1] += quant_error * 5 / 48
                img[y + 1, x] += quant_error * 7 / 48
                if x + 1 < width: img[y + 1, x + 1] += quant_error * 5 / 48
                if x + 2 < width: img[y + 1, x + 2] += quant_error * 3 / 48
            
            # Row 2 (y+2)
            if y + 2 < height:
                if x - 2 >= 0: img[y + 2, x - 2] += quant_error * 1 / 48
                if x - 1 >= 0: img[y + 2, x - 1] += quant_error * 3 / 48
                img[y + 2, x] += quant_error * 5 / 48
                if x + 1 < width: img[y + 2, x + 1] += quant_error * 3 / 48
                if x + 2 < width: img[y + 2, x + 2] += quant_error * 1 / 48
                        
    return out

@jit(nopython=True)
def _stucki_jit(img):
    height, width = img.shape
    out = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            new_pixel = 1 if old_pixel > 0.5 else 0
            out[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            # Stucki (Div 42)
            #         x   8   4
            # 2   4   8   4   2
            # 1   2   4   2   1
            
            if x + 1 < width: img[y, x + 1] += quant_error * 8 / 42
            if x + 2 < width: img[y, x + 2] += quant_error * 4 / 42
            
            if y + 1 < height:
                if x - 2 >= 0: img[y + 1, x - 2] += quant_error * 2 / 42
                if x - 1 >= 0: img[y + 1, x - 1] += quant_error * 4 / 42
                img[y + 1, x] += quant_error * 8 / 42
                if x + 1 < width: img[y + 1, x + 1] += quant_error * 4 / 42
                if x + 2 < width: img[y + 1, x + 2] += quant_error * 2 / 42
            
            if y + 2 < height:
                if x - 2 >= 0: img[y + 2, x - 2] += quant_error * 1 / 42
                if x - 1 >= 0: img[y + 2, x - 1] += quant_error * 2 / 42
                img[y + 2, x] += quant_error * 4 / 42
                if x + 1 < width: img[y + 2, x + 1] += quant_error * 2 / 42
                if x + 2 < width: img[y + 2, x + 2] += quant_error * 1 / 42
    return out

@jit(nopython=True)
def _burkes_jit(img):
    height, width = img.shape
    out = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            new_pixel = 1 if old_pixel > 0.5 else 0
            out[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            # Burkes (Div 32)
            #         x   8   4
            # 2   4   8   4   2
            
            if x + 1 < width: img[y, x + 1] += quant_error * 8 / 32
            if x + 2 < width: img[y, x + 2] += quant_error * 4 / 32
            
            if y + 1 < height:
                if x - 2 >= 0: img[y + 1, x - 2] += quant_error * 2 / 32
                if x - 1 >= 0: img[y + 1, x - 1] += quant_error * 4 / 32
                img[y + 1, x] += quant_error * 8 / 32
                if x + 1 < width: img[y + 1, x + 1] += quant_error * 4 / 32
                if x + 2 < width: img[y + 1, x + 2] += quant_error * 2 / 32
    return out

@jit(nopython=True)
def _sierra_lite_jit(img):
    height, width = img.shape
    out = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            new_pixel = 1 if old_pixel > 0.5 else 0
            out[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            # Sierra Lite (Div 4)
            #     x   2
            # 1   1
            
            if x + 1 < width: img[y, x + 1] += quant_error * 2 / 4
            
            if y + 1 < height:
                if x - 1 >= 0: img[y + 1, x - 1] += quant_error * 1 / 4
                img[y + 1, x] += quant_error * 1 / 4
    return out

class Halftoner:
    def __init__(self, image):
        """
        Initialize with a grayscale image (numpy array).
        """
        if len(image.shape) == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image
        
        self.height, self.width = self.image.shape

    def run(self, algorithm="floyd-steinberg"):
        """
        Run the specified halftoning algorithm.
        Returns a binary numpy array (0s and 1s).
        """
        # Prepare float image for error diffusion
        img_float = self.image.astype(float) / 255.0
        
        if algorithm == "floyd-steinberg":
            return _floyd_steinberg_jit(img_float)
        elif algorithm == "atkinson":
            return _atkinson_jit(img_float)
        elif algorithm == "jarvis-judice-ninke":
            return _jarvis_judice_ninke_jit(img_float)
        elif algorithm == "stucki":
            return _stucki_jit(img_float)
        elif algorithm == "burkes":
            return _burkes_jit(img_float)
        elif algorithm == "sierra-lite":
            return _sierra_lite_jit(img_float)
        elif algorithm == "bayer":
            return self.bayer_ordered()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def bayer_ordered(self):
        """
        Bayer ordered dithering (4x4 matrix).
        Vectorized implementation for speed.
        """
        # 4x4 Bayer Matrix
        bayer_matrix = np.array([
            [ 0,  8,  2, 10],
            [12,  4, 14,  6],
            [ 3, 11,  1,  9],
            [15,  7, 13,  5]
        ]) / 16.0
        
        # Tile the matrix to cover the image
        tiled_matrix = np.tile(bayer_matrix, (self.height // 4 + 1, self.width // 4 + 1))
        tiled_matrix = tiled_matrix[:self.height, :self.width]
        
        # Normalize image to 0-1
        img_norm = self.image.astype(float) / 255.0
        
        # Threshold
        out = (img_norm > tiled_matrix).astype(np.uint8)
        
        return out
