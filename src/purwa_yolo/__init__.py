"""Electronic Road Pricing: YOLO vehicle detection + licence-plate OCR.

Layers (each one only imports from the ones above it in this list):
    core/    pure logic and data classes (no Streamlit, no OpenCV)
    vision/  YOLO and plate OCR (no Streamlit)
    ui/      Streamlit widgets and rendering
    main.py  page layout and flow
"""
__version__ = "0.1.0"
