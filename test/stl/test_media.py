import os
import shutil
import pytest
import numpy as np
from PIL import Image

# Import MoviePy (handles both v1.x and v2.x directory structures)
try:
    from moviepy.editor import ColorClip, AudioClip
except ImportError:
    from moviepy import ColorClip, AudioClip

from nemantix.core import Toolset
from nemantix.stl.media_processor.base import MediaToolset

# Setup real paths for testing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "test_data_media")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_output_media")

REAL_JPG = os.path.join(INPUT_DIR, "sample.jpg")
REAL_MP4 = os.path.join(INPUT_DIR, "sample.mp4")
SILENT_MP4 = os.path.join(INPUT_DIR, "silent.mp4")
CORRUPT_FILE = os.path.join(INPUT_DIR, "corrupt.file")


@pytest.fixture(scope="session", autouse=True)
def setup_real_media():
    """Generates physical images and videos to test against before the test suite starts."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Create a real JPEG Image (1920x1080, solid blue)
    if not os.path.exists(REAL_JPG):
        img = Image.new("RGB", (1920, 1080), color="blue")
        img.save(REAL_JPG, format="JPEG")

    # 2. Create a real MP4 Video WITH audio (1 second long)
    if not os.path.exists(REAL_MP4):
        clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=1.0)
        clip.fps = 24

        # Create a simple 440Hz sine wave audio track
        def make_frame(t):
            val = np.sin(440 * 2 * np.pi * t)
            return np.vstack([val, val]).T if not np.isscalar(t) else [val, val]

        audio = AudioClip(make_frame, duration=1.0, fps=44100)
        clip = clip.with_audio(audio)
        clip.write_videofile(REAL_MP4, codec="libx264", audio_codec="aac", logger=None)

    # 3. Create a real MP4 Video WITHOUT audio (1 second long)
    if not os.path.exists(SILENT_MP4):
        silent_clip = ColorClip(size=(640, 480), color=(0, 255, 0), duration=1.0)
        silent_clip.fps = 24
        silent_clip.write_videofile(SILENT_MP4, codec="libx264", logger=None)

    # 4. Create a corrupt file for exception handling
    with open(CORRUPT_FILE, "w", encoding="utf-8") as f:
        f.write("This is just text, not a valid media file.")

    yield  # Pytest runs all tests here

    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_output_dir():
    """Empties the output directory before each test runs."""
    if os.path.exists(OUTPUT_DIR):
        for file_name in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file_name)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    yield


class TestMediaToolset:
    # --- Initialization ---

    def test_init_creates_directory(self):
        """Test that the working directory is created if it doesn't exist."""
        test_dir = os.path.join(BASE_DIR, "temp_media_dir")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        MediaToolset(working_directory=test_dir)
        assert os.path.exists(test_dir)
        shutil.rmtree(test_dir)  # Immediate cleanup

    # --- Image Tests ---

    def test_get_file_info_image(self):
        """Test retrieving metadata for a real image."""
        ts_info = Toolset.get_tool(
            tool_name="MediaToolset.get_file_info", instance_args=(OUTPUT_DIR,)
        )

        result = ts_info(REAL_JPG)

        assert "JPEG" in result
        assert "1920" in result and "1080" in result
        assert "RGB" in result

    def test_resize_image(self):
        """Test actual image resizing logic."""
        ts_resize = Toolset.get_tool(
            tool_name="MediaToolset.resize_image", instance_args=(OUTPUT_DIR,)
        )

        result = ts_resize(REAL_JPG, width=800, height=600)

        assert "800x600" in result
        output_files = [
            f
            for f in os.listdir(OUTPUT_DIR)
            if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")
        ]
        assert len(output_files) == 1

    def test_grayscale_image(self):
        """Test physical grayscale conversion."""
        ts_gray = Toolset.get_tool(
            tool_name="MediaToolset.grayscale_image", instance_args=(OUTPUT_DIR,)
        )

        result = ts_gray(REAL_JPG)

        assert "grayscale" in result.lower()
        output_files = [
            f
            for f in os.listdir(OUTPUT_DIR)
            if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")
        ]
        assert len(output_files) == 1

    # --- Video Tests ---

    def test_get_file_info_video(self):
        """Test retrieving metadata for a real video."""
        ts_info = Toolset.get_tool(
            tool_name="MediaToolset.get_file_info", instance_args=(OUTPUT_DIR,)
        )

        result = ts_info(REAL_MP4)

        assert "1.0" in result  # 1.0 seconds duration
        assert "640" in result and "480" in result  # Resolution
        assert "24" in result  # fps

    def test_trim_video(self):
        """Test actual video trimming logic."""
        ts_trim = Toolset.get_tool(
            tool_name="MediaToolset.trim_video", instance_args=(OUTPUT_DIR,)
        )

        # Trim the 1-second video down to half a second
        result = ts_trim(REAL_MP4, start_time=0.0, end_time=0.5)

        assert "trimmed" in result.lower()
        output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp4")]
        assert len(output_files) == 1

    def test_extract_audio_success(self):
        """Test physically extracting an audio track from a video."""
        ts_extract = Toolset.get_tool(
            tool_name="MediaToolset.extract_audio", instance_args=(OUTPUT_DIR,)
        )

        result = ts_extract(REAL_MP4)

        assert "extracted" in result.lower()
        # Verify an audio file was saved
        output_files = [
            f
            for f in os.listdir(OUTPUT_DIR)
            if f.endswith(".mp3") or f.endswith(".wav")
        ]
        assert len(output_files) == 1

    def test_extract_audio_no_track(self):
        """Test error handling when a video physically lacks an audio track."""
        ts_extract = Toolset.get_tool(
            tool_name="MediaToolset.extract_audio", instance_args=(OUTPUT_DIR,)
        )

        result = ts_extract(SILENT_MP4)
        assert "Error" in result and "audio" in result.lower()

    # --- Error Handling ---

    def test_file_not_found(self):
        """Test that missing files return an error immediately."""
        ts_resize = Toolset.get_tool(
            tool_name="MediaToolset.resize_image", instance_args=(OUTPUT_DIR,)
        )

        missing_file = os.path.join(INPUT_DIR, "ghost_file.jpg")
        result = ts_resize(missing_file, width=100, height=100)

        assert "Error" in result and "not found" in result.lower()

    def test_processing_exception(self):
        """Test that internal library errors are caught and reported when fed bad data."""
        ts_resize = Toolset.get_tool(
            tool_name="MediaToolset.resize_image", instance_args=(OUTPUT_DIR,)
        )

        # Handing a plain text file to Pillow should trigger an exception
        result = ts_resize(CORRUPT_FILE, width=100, height=100)

        assert "Error" in result
