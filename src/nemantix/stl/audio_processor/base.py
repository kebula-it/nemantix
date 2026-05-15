import os
import numpy as np
from moviepy import AudioFileClip, concatenate_audioclips
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

from nemantix.core import tool, Toolset


class AudioProcessorToolset(Toolset):
    """
    A toolset for audio processing using MoviePy (FFMPEG-based).
    """

    def __init__(self, working_directory: str = "./audio_output"):
        super().__init__()
        self.work_dir = working_directory
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def _get_out_path(self, filename: str) -> str:
        return os.path.join(self.work_dir, filename)

    @tool
    def get_audio_info(self, file_path: str) -> str:
        """
        Returns metadata about an audio file using FFMPEG.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            str: A formatted string containing duration, sampling rate, and channel count.

        Example call:
            get_audio_info(file_path="audio.mp3")
        """
        if not os.path.exists(file_path):
            return "Error: File not found."
        try:
            clip = AudioFileClip(file_path)
            info = (f"Duration: {clip.duration:.2f}s\n"
                    f"FPS (Sampling Rate): {clip.fps}Hz\n"
                    f"Channels: {clip.nchannels}")
            clip.close()
            return info
        except Exception as e:
            return f"Error reading audio info: {e}"

    @tool
    def convert_format(self, input_path: str, output_format: str = "mp3") -> str:
        """
        Converts an audio file to a different format (mp3, wav, m4a, etc.).

        Args:
            input_path (str): Path to the source audio file.
            output_format (str): The target file extension format. Defaults to "mp3".

        Returns:
            str: Confirmation message with the path to the converted file.

        Example call:
            convert_format(input_path="recording.wav", output_format="mp3")
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."
        try:
            clip = AudioFileClip(input_path)

            filename = os.path.splitext(os.path.basename(input_path))[0]
            out_name = f"{filename}.{output_format}"
            out_path = self._get_out_path(out_name)

            clip.write_audiofile(out_path, logger=None)
            clip.close()
            return f"Converted to {output_format}: {out_path}"
        except Exception as e:
            return f"Error converting format: {e}"

    @tool
    def trim_audio(
            self,
            input_path: str,
            start_time: float,
            end_time: float,
            output_filename: str = "trimmed.mp3",
    ) -> str:
        """
        Cuts a specific segment of audio.

        Args:
            input_path (str): Path to the source audio file.
            start_time (float): The start timestamp in seconds for the trim.
            end_time (float): The end timestamp in seconds for the trim.
            output_filename (str): The name of the resulting trimmed file. Defaults to "trimmed.mp3".

        Returns:
            str: Confirmation message with the saved file path and time range.

        Example call:
            trim_audio(input_path="song.mp3", start_time=10.0, end_time=20.5, output_filename="clip.mp3")
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."
        try:
            clip = AudioFileClip(input_path)
            clip = clip.subclipped(start_time, end_time)

            out_path = self._get_out_path(output_filename)
            clip.write_audiofile(out_path, logger=None)
            clip.close()
            return f"Trimmed audio ({start_time}-{end_time}s) saved to: {out_path}"
        except Exception as e:
            return f"Error trimming audio: {e}"

    @tool
    def adjust_volume(
            self,
            input_path: str,
            volume_factor: float,
            output_filename: str = "volume_adjusted.mp3",
    ) -> str:
        """
        Changes volume by a factor (e.g., 2.0 = double volume, 0.5 = half volume).

        Args:
            input_path (str): Path to the source audio file.
            volume_factor (float): Multiplier for the audio volume.
            output_filename (str): The name of the resulting adjusted file. Defaults to "volume_adjusted.mp3".

        Returns:
            str: Confirmation message with the volume factor applied.

        Example call:
            adjust_volume(input_path="audio.mp3", volume_factor=1.5, output_filename="louder.mp3")
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."
        try:
            clip = AudioFileClip(input_path)
            clip = clip.with_volume_scaled(volume_factor)

            out_path = self._get_out_path(output_filename)
            clip.write_audiofile(out_path, logger=None)
            clip.close()
            return f"Volume scaled by {volume_factor}x. Saved to: {out_path}"
        except Exception as e:
            return f"Error adjusting volume: {e}"

    @tool
    def merge_audios(self, file_paths: list[str], output_filename: str = "merged.mp3") -> str:
        """
        Concatenates multiple audio files sequentially.

        Args:
            file_paths (list[str]): List of strings representing paths to the audio files to be merged.
            output_filename (str): The name of the resulting merged file. Defaults to "merged.mp3".

        Returns:
            str: Confirmation message indicating the number of files merged.

        Example call:
            merge_audios(file_paths=["part1.mp3", "part2.mp3"], output_filename="full_track.mp3")
        """
        if not file_paths:
            return "Error: No file paths provided."
        try:
            clips = []
            for path in file_paths:
                if os.path.exists(path):
                    clips.append(AudioFileClip(path))
                else:
                    return f"Error: File '{path}' not found."

            final_clip = concatenate_audioclips(clips)
            out_path = self._get_out_path(output_filename)
            final_clip.write_audiofile(out_path, logger=None)

            for c in clips:
                c.close()
            final_clip.close()

            return f"Merged {len(file_paths)} files into: {out_path}"
        except Exception as e:
            return f"Error merging audio: {e}"

    @tool
    def decode_audio(
            self,
            input_path: str,
            start_time: float | None = None,
            end_time: float | None = None,
            fps: int = 44100,
    ) -> str:
        """
        Decodes audio into a NumPy array (waveform).

        Args:
            input_path (str): Path to source file.
            start_time (float, optional): Start timestamp in seconds. Defaults to None (start of file).
            end_time (float, optional): End timestamp in seconds. Defaults to None (end of file).
            fps (int, optional): Sample rate to resample to (e.g. 16000, 44100). Defaults to 44100.

        Returns:
            str: Path to the .npy file and metadata about the array shape.

        Example call:
            decode_audio(input_path="raw.wav", start_time=0.0, end_time=5.0, fps=16000)
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."

        try:
            clip = AudioFileClip(input_path)
            s_time = start_time if start_time is not None else 0
            e_time = end_time if end_time is not None else clip.duration

            clip = clip.subclipped(s_time, e_time)
            arr = clip.to_soundarray(fps=fps)

            filename = f"audio_waveform_{int(s_time)}-{int(e_time)}.npy"
            out_path = self._get_out_path(filename)

            np.save(out_path, arr)
            clip.close()

            return f"Audio decoded to NumPy: {out_path} (Shape: {arr.shape}, Rate: {fps}Hz)"

        except Exception as e:
            return f"Error decoding audio: {e}"

    @tool
    def apply_fade(
            self,
            input_path: str,
            fade_type: str = "both",
            duration: float = 1.0,
            output_filename: str = "faded.mp3",
    ) -> str:
        """
        Applies a fade-in and/or fade-out effect to the audio.

        Args:
            input_path (str): Path to the source audio.
            fade_type (str): 'in' (start), 'out' (end), or 'both'. Defaults to 'both'.
            duration (float): Duration of the fade in seconds. Defaults to 1.0.
            output_filename (str): Name of the saved file.

        Returns:
            str: Path to the processed file.
        """
        if not os.path.exists(input_path):
            return "Error: Input file not found."

        try:
            clip = AudioFileClip(input_path)
            effects_to_apply = []

            # 1. Select Effects
            if fade_type.lower() in ["in", "both"]:
                # fade in usually requires start_time=0 logic internally
                effects_to_apply.append(AudioFadeIn(duration=duration))

            if fade_type.lower() in ["out", "both"]:
                effects_to_apply.append(AudioFadeOut(duration=duration))

            # 2. Apply Effects
            if effects_to_apply:
                clip = clip.with_effects(effects_to_apply)

            # 3. Save
            out_path = self._get_out_path(output_filename)
            clip.write_audiofile(out_path, logger=None)
            clip.close()

            return f"Applied '{fade_type}' fade ({duration}s) to: {out_path}"

        except Exception as e:
            return f"Error applying fade: {e}"
