print("Starting imports...")
try:
    import numpy
    print("numpy imported")
except Exception as e:
    print(f"numpy import failed: {e}")

try:
    import streamlit
    print("streamlit imported")
except Exception as e:
    print(f"streamlit import failed: {e}")
