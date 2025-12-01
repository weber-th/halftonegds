import numpy as np
import sys
import os

# Add script directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.halftoning import Halftoner

def test_halftoning():
    print("Testing Halftoning Algorithms...")
    # Create a gradient image
    width, height = 100, 100
    image = np.zeros((height, width), dtype=np.uint8)
    for i in range(width):
        image[:, i] = int(255 * i / width)
    
    halftoner = Halftoner(image)
    
    algorithms = ["floyd-steinberg", "atkinson", "jarvis-judice-ninke", "bayer"]
    
    for algo in algorithms:
        try:
            res = halftoner.run(algo)
            assert res.shape == (height, width)
            assert res.dtype == np.uint8
            assert np.all(np.isin(res, [0, 1]))
            print(f"  [PASS] {algo}")
        except Exception as e:
            print(f"  [FAIL] {algo}: {e}")

def test_rle_logic():
    print("\nTesting RLE Optimization Logic...")
    # Create a simple row: 0 1 1 1 0 1 0
    row = np.array([0, 1, 1, 1, 0, 1, 0])
    pixel_size = 10
    
    # Expected rectangles:
    # 1. Start index 1, end index 4 (exclusive) -> x=10 to x=40
    # 2. Start index 5, end index 6 (exclusive) -> x=50 to x=60
    
    padded = np.concatenate(([0], row, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    print(f"  Row: {row}")
    print(f"  Starts: {starts}")
    print(f"  Ends:   {ends}")
    
    assert len(starts) == 2
    assert starts[0] == 1
    assert ends[0] == 4
    assert starts[1] == 5
    assert ends[1] == 6
    print("  [PASS] RLE Logic")

def test_gds_generation():
    print("\nTesting GDS Generation...")
    from src.gds_writer import GDSWriter
    import gdstk
    
    # Create a simple 10x10 binary image
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:5, 2:5] = 1
    
    writer = GDSWriter(image, pixel_size_um=1.0)
    lib = writer.generate_library()
    
    assert isinstance(lib, gdstk.Library)
    cell = lib.cells[0]
    assert len(cell.polygons) > 0
    print("  [PASS] GDS Generation")

if __name__ == "__main__":
    test_halftoning()
    test_rle_logic()
    test_gds_generation()
    print("\nVerification Complete.")
