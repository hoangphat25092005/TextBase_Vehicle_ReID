"""CLIP-based text -> vehicle search.

Import submodules directly (CLIP_Query.clip_encoder / CLIP_Query.clip_index) so that
code which only needs the index helpers does not require open_clip to be installed.
"""
