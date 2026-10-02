import sys
from types import ModuleType, SimpleNamespace

import pytest

from wasman.media import enable_current_moviepy_recorder


@pytest.mark.parametrize("count", [61, 1081, 1701, 1808])
def test_moviepy2_bridge_preserves_every_cfr_frame(monkeypatch, count):
    moviepy, recorder = ModuleType("moviepy"), ModuleType("isaaclab.envs.utils.video_recorder")
    def factory(sequence, fps=None, **kwargs):
        return SimpleNamespace(sequence=sequence, duration=sum([1 / fps] * len(sequence)))
    moviepy.ImageSequenceClip = factory
    recorder.ImageSequenceClip = None
    monkeypatch.setitem(sys.modules, "moviepy", moviepy)
    monkeypatch.setitem(sys.modules, "isaaclab.envs.utils.video_recorder", recorder)
    enable_current_moviepy_recorder()
    clip = recorder.ImageSequenceClip([None] * count, fps=30)
    assert int(clip.duration * 30) == count
    assert clip.end == clip.duration
    assert moviepy.ImageSequenceClip is factory
