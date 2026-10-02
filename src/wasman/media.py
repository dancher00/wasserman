"""Compatibility at the application boundary, without modifying vendored Isaac Lab."""


def enable_current_moviepy_recorder():
    """Bridge Isaac Lab's legacy MoviePy import to the supported MoviePy 2 API.

    Call after AppLauncher initialization and before constructing a recorder.
    Also prevent summed frame durations from dropping the last CFR frame when
    MoviePy truncates ``duration * fps`` to an integer. This changes no pixels,
    sample timestamps or upstream files.
    """
    import math
    from importlib import import_module

    from moviepy import ImageSequenceClip

    recorder = import_module("isaaclab.envs.utils.video_recorder")
    def frame_exact_clip(sequence, fps=None, **kwargs):
        clip = ImageSequenceClip(sequence, fps=fps, **kwargs)
        if fps is not None:
            clip.duration = math.nextafter(len(clip.sequence) / fps, math.inf)
            clip.end = clip.duration
        return clip

    recorder.ImageSequenceClip = frame_exact_clip
