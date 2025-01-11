"""
@author: IDGallagher
@title: IG Motion Video Search
@nickname: IG Motion Video Search
@description: Search an index of videos by motion image
"""

from .nodes import *

NODE_CLASS_MAPPINGS = {
    "IG Motion Video Search":       IG_MotionVideoSearch,
    "IG Motion Video Frame":          IG_MotionVideoFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IG Motion Video Search":       "🐂 IG Motion Video Search",
    "IG Motion Video Frame":          "🖼️ IG Motion Video Frame"
}