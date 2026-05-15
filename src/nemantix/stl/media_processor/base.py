import os

from PIL import Image, ImageOps
from moviepy import VideoFileClip

from nemantix.core import tool, Toolset
from nemantix.common.logger import get_package_logger

logger = get_package_logger(__name__)


class MediaToolset(Toolset):
    """
    A toolset for media processing (Image & Video).
    Fully compatible with MoviePy v2.0.0+.
    """

    def __init__(self, working_directory: str = "./media_output"):
        """
        Initialize the media toolset.

        Args:
            working_directory (str): Path to the directory where processed files will be saved.
        """
        super().__init__()
        self.work_dir = working_directory
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def _get_out_path(self, filename: str) -> str:
        """Helper to get full path for output files."""
        return os.path.join(self.work_dir, filename)

    @tool
    def get_file_info(self, file_path: str) -> str:
        """
        Returns metadata about a media file (resolution, duration, format).
        Supports both images and videos.

        Args:
            file_path (str): The absolute or relative path to the media file.

        Returns:
            str: A summary of the file's metadata (dimensions, fps, format, etc.).

        Example call:
            get_file_info(
                file_path="inputs/vacation.mp4"
            )
        """
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found."

        # Try Image
        try:
            with Image.open(file_path) as img:
                return (f"Image Info:\nFormat: {img.format}\n"
                        f"Size: {img.size} (WxH)\nMode: {img.mode}")
        except IOError:
            pass

        # Try Video
        try:
            clip = VideoFileClip(file_path)
            info = (f"Video Info:\nDuration: {clip.duration}s\n"
                    f"Resolution: {clip.size}\nFPS: {clip.fps}")
            clip.close()
            return info
        except Exception:
            pass

        return "Unknown media type or unreadable file."

    # --- IMAGE TOOLS (Pillow-based) ---

    @tool
    def resize_image(
            self,
            input_path: str,
            width: int,
            height: int,
            output_filename: str = "resized.png",
    ) -> str:
        """
        Resizes an image to specific dimensions.

        Args:
            input_path (str): Path to the source image.
            width (int): The target width in pixels.
            height (int): The target height in pixels.
            output_filename (str, optional): Name of the saved file. Defaults to "resized.png".

        Returns:
            str: Success message with the path to the output file.

        Example call:
            resize_image(
                input_path="photo.jpg",
                width=800,
                height=600,
                output_filename="thumbnail.jpg"
            )
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."
        try:
            with Image.open(input_path) as img:
                img = img.resize((width, height))
                out_path = self._get_out_path(output_filename)
                img.save(out_path)
                return f"Image resized to {width}x{height} and saved to: {out_path}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def grayscale_image(self, input_path: str, output_filename: str = "bw.png") -> str:
        """
        Converts an image to black and white (grayscale).

        Args:
            input_path (str): Path to the source image.
            output_filename (str, optional): Name of the saved file. Defaults to "bw.png".

        Returns:
            str: Success message with the path to the output file.

        Example call:
            grayscale_image(
                input_path="portrait.jpg",
                output_filename="portrait_bw.jpg"
            )
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."
        try:
            with Image.open(input_path) as img:
                img = ImageOps.grayscale(img)
                out_path = self._get_out_path(output_filename)
                img.save(out_path)
                return f"Image converted to grayscale and saved to: {out_path}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def rotate_image(self,
                     input_path: str,
                     degrees: int = 90,
                     output_filename: str = "rotated.png") -> str:
        """
        Rotates an image by a specified number of degrees.

        Args:
            input_path (str): Path to the source image.
            degrees (int, optional): Degrees to rotate (counter-clockwise). Defaults to 90.
            output_filename (str, optional): Name of the saved file. Defaults to "rotated.png".

        Returns:
            str: Success message with the path to the output file.

        Example call:
            rotate_image(
                input_path="scan.jpg",
                degrees=180,
                output_filename="scan_fixed.jpg"
            )
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."
        try:
            with Image.open(input_path) as img:
                img = img.rotate(degrees, expand=True)
                out_path = self._get_out_path(output_filename)
                img.save(out_path)
                return f"Image rotated {degrees} degrees and saved to: {out_path}"
        except Exception as e:
            return f"Error: {e}"

    # --- VIDEO TOOLS (MoviePy v2.0+) ---

    @tool
    def trim_video(
            self,
            input_path: str,
            start_time: float,
            end_time: float,
            output_filename: str = "trimmed.mp4",
    ) -> str:
        """
        Cuts a video clip from start_time to end_time.

        Args:
            input_path (str): Path to the source video.
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
            output_filename (str, optional): Name of the saved file. Defaults to "trimmed.mp4".

        Returns:
            str: Success message with the path to the output file.

        Example call:
            trim_video(
                input_path="raw_footage.mp4",
                start_time=10.5,
                end_time=25.0,
                output_filename="highlight.mp4"
            )
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."
        try:
            clip = VideoFileClip(input_path)

            # v2.0 Change: Use 'subclipped' (past tense)
            clip = clip.subclipped(start_time, end_time)

            out_path = self._get_out_path(output_filename)
            clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
            clip.close()
            return f"Video trimmed ({start_time}-{end_time}s) and saved to: {out_path}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def crop_video(
            self,
            input_path: str,
            x1: int,
            y1: int,
            width: int,
            height: int,
            output_filename: str = "cropped.mp4",
    ) -> str:
        """
        Crops a video to a specific rectangular area.

        Args:
            input_path (str): Path to the source video.
            x1 (int): The x-coordinate of the top-left corner of the crop box.
            y1 (int): The y-coordinate of the top-left corner of the crop box.
            width (int): The width of the cropped area.
            height (int): The height of the cropped area.
            output_filename (str, optional): Name of the saved file. Defaults to "cropped.mp4".

        Returns:
            str: Success message with the path to the output file.

        Example call:
            crop_video(
                input_path="screen_recording.mp4",
                x1=0,
                y1=0,
                width=1920,
                height=1080
            )
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."
        try:
            clip = VideoFileClip(input_path)

            # v2.0 Change: Use 'cropped' (past tense) directly on the clip
            clip = clip.cropped(x1=x1, y1=y1, width=width, height=height)

            out_path = self._get_out_path(output_filename)
            clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
            clip.close()
            return f"Video cropped to {width}x{height} at ({x1},{y1}) and saved to: {out_path}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def resize_video(
            self,
            input_path: str,
            width: int,
            height: int,
            output_filename: str = "resized_video.mp4",
    ) -> str:
        """
        Resizes a video to specific dimensions.

        Args:
            input_path (str): Path to the source video.
            width (int): Target width in pixels.
            height (int): Target height in pixels.
            output_filename (str, optional): Name of the saved file. Defaults to "resized_video.mp4".

        Returns:
            str: Success message with the path to the output file.

        Example call:
            resize_video(
                input_path="input_hd.mp4",
                width=1280,
                height=720,
                output_filename="output_720p.mp4"
            )
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."
        try:
            clip = VideoFileClip(input_path)

            # v2.0 Change: Use 'resized' (past tense) directly on the clip
            clip = clip.resized(width=width, height=height)

            out_path = self._get_out_path(output_filename)
            clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
            clip.close()
            return f"Video resized and saved to: {out_path}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def extract_audio(self, input_path: str, output_filename: str = "audio.mp3") -> str:
        """
        Extracts the audio track from a video and saves it as an MP3.

        Args:
            input_path (str): Path to the source video.
            output_filename (str, optional): Name of the saved audio file. Defaults to "audio.mp3".

        Returns:
            str: Success message with the path to the output file.

        Example call:
            extract_audio(
                input_path="music_video.mp4",
                output_filename="track.mp3"
            )
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."
        try:
            clip = VideoFileClip(input_path)
            out_path = self._get_out_path(output_filename)
            if clip.audio:
                clip.audio.write_audiofile(out_path)
                clip.close()
                return f"Audio extracted and saved to: {out_path}"
            else:
                clip.close()
                return "Error: No audio track found in video."
        except Exception as e:
            return f"Error: {e}"

    @tool
    def extract_frames(
            self,
            input_path: str,
            start_time: float | None = None,
            end_time: float | None = None,
            fps: float | None = None,
            output_format: str = "files",
    ) -> str:
        """
        Decodes a video into frames. If start/end times are omitted, processes the entire video.

        Args:
            input_path (str): Path to the source video.
            start_time (float, optional): Start in seconds. If None, starts at 0.
            end_time (float, optional): End in seconds. If None, goes to end of video.
            fps (float, optional): Frames per second.
                                   If None, uses the video's native FPS (extracts all frames).
                                   If provided (e.g., 0.5), it resamples (skips or duplicates frames).
            output_format (str, optional): "files" (default) or "numpy".

        Returns:
            str: Path to the output directory or .npy file.
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."

        try:
            clip = VideoFileClip(input_path)

            # 1. Handle Time Defaults
            s_time = start_time if start_time is not None else 0
            e_time = end_time if end_time is not None else clip.duration

            # 2. Handle FPS Defaults
            # If fps is None, iter_frames uses the clip's native fps automatically.
            # We explicitly grab it here just for logging/checking purposes.
            effective_fps = fps if fps is not None else clip.fps

            # Safety Check: Warn if upsampling significantly (e.g. asking 60fps from 24fps source)
            if fps is not None and fps > clip.fps:
                logger.warning(f"Requested FPS ({fps}) > Source FPS ({clip.fps}). "
                               "Output will contain duplicate frames.")

            # Subclip (MoviePy v2.0+ uses 'subclipped')
            clip = clip.subclipped(s_time, e_time)

            # 3. Output Handling
            if output_format == "numpy":
                import numpy as np

                frames = [frame for frame in clip.iter_frames(fps=effective_fps)]

                if not frames:
                    return "Error: No frames extracted."

                arr = np.array(frames)

                filename = f"frames_{int(s_time)}-{int(e_time)}.npy"
                out_path = self._get_out_path(filename)
                np.save(out_path, arr)

                clip.close()
                return f"Frames saved to {out_path} (Shape: {arr.shape})"

            else:  # "files"
                filename_clean = os.path.splitext(os.path.basename(input_path))[0]
                dir_suffix = f"frames_{int(s_time)}-{int(e_time)}"
                output_dir = self._get_out_path(f"{filename_clean}_{dir_suffix}")

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                count = 0
                for frame in clip.iter_frames(fps=effective_fps):
                    img = Image.fromarray(frame)
                    img.save(os.path.join(output_dir, f"frame_{count:04d}.png"))
                    count += 1

                clip.close()
                return f"Extracted {count} frames to: {output_dir}"

        except Exception as e:
            return f"Error: {e}"
