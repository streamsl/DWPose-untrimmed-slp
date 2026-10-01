"""The detector and pose model configs, shipped as package data (paths.CONFIG_DIR).

Byte-identical copies of the YOLOX-L (mmdet) and DWPose-L 384x288 (mmpose) configs; their sha256
is part of the extraction hash, so they must never be edited. They are read by file path with
mmengine's Config.fromfile and are not meant to be imported.
"""
