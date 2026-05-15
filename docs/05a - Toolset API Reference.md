## Toolset API Reference

This document provide the API reference for both the base Toolset class and all the Toolsets provided in the Nemantix Standard Toolset Library (NSTL).

### Available Toolsets Summary

#### Core Framework
* **[`Toolset`](#class-toolset)**: Initializes the base Toolset, specifically setting up a common state dictionary and dynamic tool instantiation across all inheriting tools.

#### File Operations
* **[`LocalFileSystemToolset`](#class-localfilesystemtoolset)**: Safe file system operations within a sandboxed directory.
* **[`RemoteFileSystemToolset`](#class-remotefilesystemtoolset)**: Interacting with remote file servers via FTP, FTPS, or SFTP.

#### Communication & Messaging
* **[`EmailToolset`](#class-emailtoolset)**: Sending and reading emails via SMTP and IMAP.
* **[`MessagingToolset`](#class-messagingtoolset)**: Interacting with Telegram Bots to send messages.

#### Web & Networking
* **[`RequestsToolset`](#class-requeststoolset)**: Performing stateless HTTP requests with explicit authentication.
* **[`WebSearchToolset`](#class-websearchtoolset)**: Searching the live web and news without requiring an API key.

#### Data & Computation
* **[`MathSolverToolset`](#class-mathsolvertoolset)**: Advanced symbolic mathematical calculations using the SymPy library.
* **[`SqlExplorerToolset`](#class-sqlexplorertoolset)**: Schema inspection and query execution on SQL databases using SQLAlchemy.

#### Media Processing
* **[`AudioProcessorToolset`](#audioprocessortoolset)**: Audio processing using MoviePy (FFMPEG-based).
* **[`MediaToolset`](#class-mediatoolset)**: Media processing for images and videos.


---
## <kbd>class</kbd> `Toolset`

<a href="../../../../src/nemantix/core/runtime.py#L659"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__()

```

Initializes the base Toolset, specifically setting up a common `self.state` dictionary across tools.

---

<a href="../../../../src/nemantix/core/runtime.py#L714"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `get_instance`

```python
get_instance(target_class, alias: str = None, args=None)

```

Returns an existing instance from cache or creates a new one.

---

<a href="../../../../src/nemantix/core/runtime.py#L705"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `get_registered_classes`

```python
get_registered_classes() → list

```

Retrieves a list of all classes that have been registered in the toolset registry.

---

<a href="../../../../src/nemantix/core/runtime.py#L743"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_tool`

```python
get_tool(tool_name: str, instance_alias: str = None, instance_args=None)

```

Retrieves a tool by name.

---

<a href="../../../../src/nemantix/core/runtime.py#L695"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `get_tool_descriptions`

```python
get_tool_descriptions() → dict

```

Retrieves the docstrings for the callee class's tools.

---

### <kbd>classmethod</kbd> `get_tool_parameters`

```python
get_tool_parameters(tool_name: str = None) → dict

```

Retrieves the parameter definitions for the callee class's tools.

---

### <kbd>classmethod</kbd> `get_tools`

```python
get_tools() → list[Callable]

```

Retrieves the list of tools for the callee class.

---

<a href="../../../../src/nemantix/core/runtime.py#L684"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `register_alias`

```python
register_alias(tool_class: str, tool_name: str, alias: str) → bool

```

Registers an alias for a specific tool within a tool class, making it accessible via the new alias name.

---

<a href="../../../../src/nemantix/core/runtime.py#L681"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `reset_state`

```python
reset_state()

```

Clears the shared state dictionary for the toolset instance.

---

<a href="../../../../src/nemantix/core/runtime.py#L757"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `run_tool`

```python
run_tool(
    tool_name: str,
    *args,
    instance_alias: str = None,
    instance_args=None,
    **kwargs
)

```

Executes a tool by name.

---

<a href="../../../../src/nemantix/core/runtime.py#L678"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `update_state`

```python
update_state(**kwargs)
```

Updates the shared state dictionary with the provided keyword arguments.

---

<a id="audioprocessortoolset"></a>
## <kbd>class</kbd> `AudioProcessorToolset`
A toolset for audio processing using MoviePy (FFMPEG-based).

<a href="../../../../../src/nemantix/stl/audio_processor/base.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(working_directory: str = './audio_output')
```








---

<a href="../../../../../src/nemantix/stl/audio_processor/base/py/adjust_volume#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `adjust_volume`

```python
adjust_volume(
    input_path: str,
    volume_factor: float,
    output_filename: str = 'volume_adjusted.mp3'
) → str
```

Changes volume by a factor (e.g., 2.0 = double volume, 0.5 = half volume).



**Args:**

 - <b>`input_path`</b> (str):  Path to the source audio file.
 - <b>`volume_factor`</b> (float):  Multiplier for the audio volume.
 - <b>`output_filename`</b> (str):  The name of the resulting adjusted file. Defaults to "volume_adjusted.mp3".



**Returns:**

 - <b>`str`</b>:  Confirmation message with the volume factor applied.

Example call: adjust_volume(input_path="audio.mp3", volume_factor=1.5, output_filename="louder.mp3")

---

<a href="../../../../../src/nemantix/stl/audio_processor/base/py/apply_fade#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `apply_fade`

```python
apply_fade(
    input_path: str,
    fade_type: str = 'both',
    duration: float = 1.0,
    output_filename: str = 'faded.mp3'
) → str
```

Applies a fade-in and/or fade-out effect to the audio.



**Args:**

 - <b>`input_path`</b> (str):  Path to the source audio.
 - <b>`fade_type`</b> (str):  'in' (start), 'out' (end), or 'both'. Defaults to 'both'.
 - <b>`duration`</b> (float):  Duration of the fade in seconds. Defaults to 1.0.
 - <b>`output_filename`</b> (str):  Name of the saved file.



**Returns:**

 - <b>`str`</b>:  Path to the processed file.

---

<a href="../../../../../src/nemantix/stl/audio_processor/base/py/convert_format#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `convert_format`

```python
convert_format(input_path: str, output_format: str = 'mp3') → str
```

Converts an audio file to a different format (mp3, wav, m4a, etc.).



**Args:**

 - <b>`input_path`</b> (str):  Path to the source audio file.
 - <b>`output_format`</b> (str):  The target file extension format. Defaults to "mp3".



**Returns:**

 - <b>`str`</b>:  Confirmation message with the path to the converted file.

Example call: convert_format(input_path="recording.wav", output_format="mp3")

---

<a href="../../../../../src/nemantix/stl/audio_processor/base/py/decode_audio#L170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `decode_audio`

```python
decode_audio(
    input_path: str,
    start_time: float = None,
    end_time: float = None,
    fps: int = 44100
) → str
```

Decodes audio into a NumPy array (waveform).



**Args:**

 - <b>`input_path`</b> (str):  Path to source file.
 - <b>`start_time`</b> (float, optional):  Start timestamp in seconds. Defaults to None (start of file).
 - <b>`end_time`</b> (float, optional):  End timestamp in seconds. Defaults to None (end of file).
 - <b>`fps`</b> (int, optional):  Sample rate to resample to (e.g. 16000, 44100). Defaults to 44100.



**Returns:**

 - <b>`str`</b>:  Path to the .npy file and metadata about the array shape.

Example call: decode_audio(input_path="raw.wav", start_time=0.0, end_time=5.0, fps=16000)

---

<a href="../../../../../src/nemantix/stl/audio_processor/base/py/get_audio_info#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_audio_info`

```python
get_audio_info(file_path: str) → str
```

Returns metadata about an audio file using FFMPEG.



**Args:**

 - <b>`file_path`</b> (str):  Path to the audio file.



**Returns:**

 - <b>`str`</b>:  A formatted string containing duration, sampling rate, and channel count.

Example call: get_audio_info(file_path="audio.mp3")

---

<a href="../../../../../src/nemantix/stl/audio_processor/base/py/merge_audios#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `merge_audios`

```python
merge_audios(file_paths: list[str], output_filename: str = 'merged.mp3') → str
```

Concatenates multiple audio files sequentially.



**Args:**

 - <b>`file_paths`</b> (list[str]):  List of strings representing paths to the audio files to be merged.
 - <b>`output_filename`</b> (str):  The name of the resulting merged file. Defaults to "merged.mp3".



**Returns:**

 - <b>`str`</b>:  Confirmation message indicating the number of files merged.

Example call: merge_audios(file_paths=["part1.mp3", "part2.mp3"], output_filename="full_track.mp3")

---

<a href="../../../../../src/nemantix/stl/audio_processor/base/py/trim_audio#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `trim_audio`

```python
trim_audio(
    input_path: str,
    start_time: float,
    end_time: float,
    output_filename: str = 'trimmed.mp3'
) → str
```

Cuts a specific segment of audio.



**Args:**

 - <b>`input_path`</b> (str):  Path to the source audio file.
 - <b>`start_time`</b> (float):  The start timestamp in seconds for the trim.
 - <b>`end_time`</b> (float):  The end timestamp in seconds for the trim.
 - <b>`output_filename`</b> (str):  The name of the resulting trimmed file. Defaults to "trimmed.mp3".



**Returns:**

 - <b>`str`</b>:  Confirmation message with the saved file path and time range.

Example call: trim_audio(input_path="song.mp3", start_time=10.0, end_time=20.5, output_filename="clip.mp3")




---

<a id="mediatoolset"></a>
## <kbd>class</kbd> `MediaToolset`
A toolset for media processing (Image & Video). Fully compatible with MoviePy v2.0.0+.

<a href="../../../../../src/nemantix/stl/media_processor/base.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(working_directory: str = './media_output')
```

Initialize the media toolset.



**Args:**

 - <b>`working_directory`</b> (str):  Path to the directory where processed files will be saved.




---

<a href="../../../../../src/nemantix/stl/media_processor/base/py/crop_video#L199"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `crop_video`

```python
crop_video(
    input_path: str,
    x1: int,
    y1: int,
    width: int,
    height: int,
    output_filename: str = 'cropped.mp4'
) → str
```

Crops a video to a specific rectangular area.



**Args:**

 - <b>`input_path`</b> (str):  Path to the source video.
 - <b>`x1`</b> (int):  The x-coordinate of the top-left corner of the crop box.
 - <b>`y1`</b> (int):  The y-coordinate of the top-left corner of the crop box.
 - <b>`width`</b> (int):  The width of the cropped area.
 - <b>`height`</b> (int):  The height of the cropped area.
 - <b>`output_filename`</b> (str, optional):  Name of the saved file. Defaults to "cropped.mp4".



**Returns:**

 - <b>`str`</b>:  Success message with the path to the output file.

Example call: crop_video(  input_path="screen_recording.mp4",  x1=0,  y1=0,  width=1920,  height=1080 )

---

<a href="../../../../../src/nemantix/stl/media_processor/base/py/extract_audio#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `extract_audio`

```python
extract_audio(input_path: str, output_filename: str = 'audio.mp3') → str
```

Extracts the audio track from a video and saves it as an MP3.



**Args:**

 - <b>`input_path`</b> (str):  Path to the source video.
 - <b>`output_filename`</b> (str, optional):  Name of the saved audio file. Defaults to "audio.mp3".



**Returns:**

 - <b>`str`</b>:  Success message with the path to the output file.

Example call: extract_audio(  input_path="music_video.mp4",  output_filename="track.mp3" )

---

<a href="../../../../../src/nemantix/stl/media_processor/base/py/extract_frames#L307"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `extract_frames`

```python
extract_frames(
    input_path: str,
    start_time: float = None,
    end_time: float = None,
    fps: float = None,
    output_format: str = 'files'
) → str
```

Decodes a video into frames. If start/end times are omitted, processes the entire video.



**Args:**

 - <b>`input_path`</b> (str):  Path to the source video.
 - <b>`start_time`</b> (float, optional):  Start in seconds. If None, starts at 0.
 - <b>`end_time`</b> (float, optional):  End in seconds. If None, goes to end of video.
 - <b>`fps`</b> (float, optional):  Frames per second.  If None, uses the video's native FPS (extracts all frames).  If provided (e.g., 0.5), it resamples (skips or duplicates frames).
 - <b>`output_format`</b> (str, optional):  "files" (default) or "numpy".



**Returns:**

 - <b>`str`</b>:  Path to the output directory or .npy file.

---

<a href="../../../../../src/nemantix/stl/media_processor/base/py/get_file_info#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_file_info`

```python
get_file_info(file_path: str) → str
```

Returns metadata about a media file (resolution, duration, format). Supports both images and videos.



**Args:**

 - <b>`file_path`</b> (str):  The absolute or relative path to the media file.



**Returns:**

 - <b>`str`</b>:  A summary of the file's metadata (dimensions, fps, format, etc.).

Example call: get_file_info(  file_path="inputs/vacation.mp4" )

---

<a href="../../../../../src/nemantix/stl/media_processor/base/py/grayscale_image#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `grayscale_image`

```python
grayscale_image(input_path: str, output_filename: str = 'bw.png') → str
```

Converts an image to black and white (grayscale).



**Args:**

 - <b>`input_path`</b> (str):  Path to the source image.
 - <b>`output_filename`</b> (str, optional):  Name of the saved file. Defaults to "bw.png".



**Returns:**

 - <b>`str`</b>:  Success message with the path to the output file.

Example call: grayscale_image(  input_path="portrait.jpg",  output_filename="portrait_bw.jpg" )

---

<a href="../../../../../src/nemantix/stl/media_processor/base/py/resize_image#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `resize_image`

```python
resize_image(
    input_path: str,
    width: int,
    height: int,
    output_filename: str = 'resized.png'
) → str
```

Resizes an image to specific dimensions.



**Args:**

 - <b>`input_path`</b> (str):  Path to the source image.
 - <b>`width`</b> (int):  The target width in pixels.
 - <b>`height`</b> (int):  The target height in pixels.
 - <b>`output_filename`</b> (str, optional):  Name of the saved file. Defaults to "resized.png".



**Returns:**

 - <b>`str`</b>:  Success message with the path to the output file.

Example call: resize_image(  input_path="photo.jpg",  width=800,  height=600,  output_filename="thumbnail.jpg" )

---

<a href="../../../../../src/nemantix/stl/media_processor/base/py/resize_video#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `resize_video`

```python
resize_video(
    input_path: str,
    width: int,
    height: int,
    output_filename: str = 'resized_video.mp4'
) → str
```

Resizes a video to specific dimensions.



**Args:**

 - <b>`input_path`</b> (str):  Path to the source video.
 - <b>`width`</b> (int):  Target width in pixels.
 - <b>`height`</b> (int):  Target height in pixels.
 - <b>`output_filename`</b> (str, optional):  Name of the saved file. Defaults to "resized_video.mp4".



**Returns:**

 - <b>`str`</b>:  Success message with the path to the output file.

Example call: resize_video(  input_path="input_hd.mp4",  width=1280,  height=720,  output_filename="output_720p.mp4" )

---

<a href="../../../../../src/nemantix/stl/media_processor/base/py/rotate_image#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `rotate_image`

```python
rotate_image(
    input_path: str,
    degrees: int = 90,
    output_filename: str = 'rotated.png'
) → str
```

Rotates an image by a specified number of degrees.



**Args:**

 - <b>`input_path`</b> (str):  Path to the source image.
 - <b>`degrees`</b> (int, optional):  Degrees to rotate (counter-clockwise). Defaults to 90.
 - <b>`output_filename`</b> (str, optional):  Name of the saved file. Defaults to "rotated.png".



**Returns:**

 - <b>`str`</b>:  Success message with the path to the output file.

Example call: rotate_image(  input_path="scan.jpg",  degrees=180,  output_filename="scan_fixed.jpg" )

---

<a href="../../../../../src/nemantix/stl/media_processor/base/py/trim_video#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `trim_video`

```python
trim_video(
    input_path: str,
    start_time: float,
    end_time: float,
    output_filename: str = 'trimmed.mp4'
) → str
```

Cuts a video clip from start_time to end_time.



**Args:**

 - <b>`input_path`</b> (str):  Path to the source video.
 - <b>`start_time`</b> (float):  Start time in seconds.
 - <b>`end_time`</b> (float):  End time in seconds.
 - <b>`output_filename`</b> (str, optional):  Name of the saved file. Defaults to "trimmed.mp4".



**Returns:**

 - <b>`str`</b>:  Success message with the path to the output file.

Example call: trim_video(  input_path="raw_footage.mp4",  start_time=10.5,  end_time=25.0,  output_filename="highlight.mp4" )




---

<a id="websearchtoolset"></a>
## <kbd>class</kbd> `WebSearchToolset`
A toolset for searching the live web. Does not require an API key.


**Args:**
 - <b>`region`</b> (str):  Region code (e.g., 'us-en', 'uk-en', 'wt-wt' for world). Defaults to "wt-wt".

<a href="../../../../src/nemantix/stl/web_search/base.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(region: str = 'wt-wt')
```

---

<a href="../../../../src/nemantix/stl/base/py/search_news#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `search_news`

```python
search_news(
 query: str, max_results: int = 5, backend: str = 'auto') → List[Dict[str, str]]
```

Searches strictly for news articles with adjustable result limits.



**Args:**
 - <b>`query`</b> (str):  The topic or current event to search for.
 - <b>`max_results`</b> (int, optional):  Number of results to return. Defaults to 5.
 - <b>`backend`</b> (str, optional):  The search backend to use. Defaults to 'auto'.

**Returns:**
 - <b>`List[Dict[str, str]]`</b>:  A list of dictionaries containing title, link, source, date, and snippet.

Example call: search_news(  query="renewable energy breakthroughs",  max_results=10 )


<a href="../../../../src/nemantix/stl/base/py/search_web#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `search_web`

```python
search_web(
 query: str, max_results: int = 5, backend: str = 'auto') → List[Dict[str, str]]
```

Performs a general web search with adjustable result limits.



**Args:**
 - <b>`query`</b> (str):  The search terms or question to look up.
 - <b>`max_results`</b> (int, optional):  Number of results to return. Defaults to 5.
 - <b>`backend`</b> (str, optional):  The search backend to use. Defaults to 'auto'.

**Returns:**
 - <b>`List[Dict[str, str]]`</b>:  A list of dictionaries containing title, link, and snippet.

Example call: search_web(  query="Python Metaclasses",  max_results=3 )

---

<a id="remotefilesystemtoolset"></a>
## <kbd>class</kbd> `RemoteFileSystemToolset`
A unified toolset for interacting with remote file servers via FTP, FTPS, or SFTP. Supports file transfer, internal moves, and directory management with persistent connections.

<a href="../../../../../src/nemantix/stl/remote_filesystem/base.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    host: str,
    user: str,
    password: str = '',
    port: int = None,
    protocol: str = 'ftp',
    timeout: int = 30
)
```








---

<a href="../../../../../src/nemantix/stl/remote_filesystem/base/py/create_directory#L282"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `create_directory`

```python
create_directory(directory_path: str) → str
```

Create a new directory on the remote server.



**Args:**

 - <b>`directory_path`</b> (str):  The path of the new directory to create.



**Returns:**

 - <b>`str`</b>:  Confirmation message.

Example call: create_directory(  directory_path="/uploads/2023" )

---

<a href="../../../../../src/nemantix/stl/remote_filesystem/base/py/delete_directory#L310"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `delete_directory`

```python
delete_directory(directory_path: str) → str
```

Remove a directory from the remote server. Note: The directory usually must be empty.



**Args:**

 - <b>`directory_path`</b> (str):  The path of the directory to remove.



**Returns:**

 - <b>`str`</b>:  Confirmation message.

Example call: delete_directory(  directory_path="/uploads/temp" )

---

<a href="../../../../../src/nemantix/stl/remote_filesystem/base/py/delete_file#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `delete_file`

```python
delete_file(file_path: str) → str
```

Delete a specific file from the remote server.



**Args:**

 - <b>`file_path`</b> (str):  The full path of the file to delete.



**Returns:**

 - <b>`str`</b>:  Confirmation message.

Example call: delete_file(  file_path="/temp/cache_dump.tmp" )

---

<a href="../../../../../src/nemantix/stl/remote_filesystem/base/py/download_file#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `download_file`

```python
download_file(remote_path: str, local_path: str) → str
```

Download a file from the remote server to the local file system.



**Args:**

 - <b>`remote_path`</b> (str):  The path to the file on the remote server.
 - <b>`local_path`</b> (str):  The destination path on the local machine.



**Returns:**

 - <b>`str`</b>:  Confirmation message.

Example call: download_file(  remote_path="/var/www/logs/error.log",  local_path="./logs/server_error.log" )

---

<a href="../../../../../src/nemantix/stl/remote_filesystem/base/py/list_files#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `list_files`

```python
list_files(directory: str = '.') → str
```

List files and directories in the specified path on the remote server.



**Args:**

 - <b>`directory`</b> (str):  The directory path to list. Defaults to current directory ".".



**Returns:**

 - <b>`str`</b>:  A newline-separated string of file names.

---

<a href="../../../../../src/nemantix/stl/remote_filesystem/base/py/move_file#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `move_file`

```python
move_file(source_path: str, destination_path: str) → str
```

Move or rename a file/directory internally on the remote server.



**Args:**

 - <b>`source_path`</b> (str):  The current path of the file/directory.
 - <b>`destination_path`</b> (str):  The new path or name.



**Returns:**

 - <b>`str`</b>:  Confirmation message.

Example call: move_file(  source_path="/uploads/temp.txt",  destination_path="/processed/final.txt" )

---

<a href="../../../../../src/nemantix/stl/remote_filesystem/base/py/read_file#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_file`

```python
read_file(file_path: str) → str
```

Download and read the content of a text file from the remote server directly into memory.



**Args:**

 - <b>`file_path`</b> (str):  The full path to the file on the server.



**Returns:**

 - <b>`str`</b>:  The content of the file decoded as UTF-8.

---

<a href="../../../../../src/nemantix/stl/remote_filesystem/base/py/upload_file#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `upload_file`

```python
upload_file(local_path: str, remote_path: str) → str
```

Upload a file from the local file system to the remote server.



**Args:**

 - <b>`local_path`</b> (str):  The path to the file on the local machine.
 - <b>`remote_path`</b> (str):  The destination path on the remote server.



**Returns:**

 - <b>`str`</b>:  Confirmation message.

Example call: upload_file(  local_path="./data/report.csv",  remote_path="/var/www/uploads/report.csv" )

---

<a href="../../../../../src/nemantix/stl/remote_filesystem/base/py/write_file#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `write_file`

```python
write_file(file_path: str, content: str) → str
```

Upload text content directly to a file on the remote server. Overwrites if exists.



**Args:**

 - <b>`file_path`</b> (str):  The destination path on the server including the filename.
 - <b>`content`</b> (str):  The text content to write.



**Returns:**

 - <b>`str`</b>:  Confirmation message.



---

<a id="localfilesystemtoolset"></a>
## <kbd>class</kbd> `LocalFileSystemToolset`
A Toolset for safe file system operations within a sandboxed directory. Enforces that all operations occur strictly within 'root_dir'.

<a href="../../../../../src/nemantix/stl/local_filesystem/base.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(root_dir: str)
```

Initialize the toolset with a sandbox root directory.



**Args:**

 - <b>`root_dir`</b> (str):  The absolute path to the directory where operations are allowed.




---

<a href="../../../../../src/nemantix/stl/local_filesystem/base/py/create_directory#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `create_directory`

```python
create_directory(directory_path: str) → str
```

Create a new directory. Creates intermediate parent directories if they don't exist.



**Args:**

 - <b>`directory_path`</b> (str):  The path of the directory to create.



**Returns:**

 - <b>`str`</b>:  Success message.

Example call: create_directory(  directory_path="projects/python/src" )

---

<a href="../../../../../src/nemantix/stl/local_filesystem/base/py/delete_directory#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `delete_directory`

```python
delete_directory(directory_path: str) → str
```

Recursively delete a directory and all its contents.



**Args:**

 - <b>`directory_path`</b> (str):  The path to the directory to remove.



**Returns:**

 - <b>`str`</b>:  Success or error message.

Example call: delete_directory(  directory_path="temp_build_files" )

---

<a href="../../../../../src/nemantix/stl/local_filesystem/base/py/delete_file#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `delete_file`

```python
delete_file(file_path: str) → str
```

Permanently delete a file.



**Args:**

 - <b>`file_path`</b> (str):  The path to the file to delete.



**Returns:**

 - <b>`str`</b>:  Success or error message.

Example call: delete_file(  file_path="cache/temp.log" )

---

<a href="../../../../../src/nemantix/stl/local_filesystem/base/py/get_file_info#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_file_info`

```python
get_file_info(file_path: str) → str
```

Get metadata about a file (size, modification time).



**Args:**

 - <b>`file_path`</b> (str):  The path to the file.



**Returns:**

 - <b>`str`</b>:  Metadata string including file size.

Example call: get_file_info(  file_path="logs/error.log" )

---

<a href="../../../../../src/nemantix/stl/local_filesystem/base/py/list_files#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `list_files`

```python
list_files(directory: str = '.') → str
```

List files and subdirectories in a given directory (relative to root).



**Args:**

 - <b>`directory`</b> (str, optional):  The directory to list. Defaults to "." (root).



**Returns:**

 - <b>`str`</b>:  A formatted list of files and directories, or an error message.

Example call: list_files(  directory="documents/reports" )

---

<a href="../../../../../src/nemantix/stl/local_filesystem/base/py/move_file#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `move_file`

```python
move_file(src_path: str, dst_path: str) → str
```

Move or rename a file or directory. Fails if the destination already exists.



**Args:**

 - <b>`src_path`</b> (str):  The current path of the file/directory.
 - <b>`dst_path`</b> (str):  The new path or name.



**Returns:**

 - <b>`str`</b>:  Success or error message.

Example call: move_file(  src_path="temp_data.txt",  dst_path="archive/data_2023.txt" )

---

<a href="../../../../../src/nemantix/stl/local_filesystem/base/py/read_file#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_file`

```python
read_file(file_path: str) → str
```

Read the contents of a text file (UTF-8).



**Args:**

 - <b>`file_path`</b> (str):  The path to the file to read.



**Returns:**

 - <b>`str`</b>:  The content of the file, or an error message.

Example call: read_file(  file_path="config/settings.json" )

---

<a href="../../../../../src/nemantix/stl/local_filesystem/base/py/replace_file#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `replace_file`

```python
replace_file(src_path: str, dst_path: str) → str
```

Atomically replace the destination file with the source file. Useful for safely updating a file.



**Args:**

 - <b>`src_path`</b> (str):  The path to the new file version.
 - <b>`dst_path`</b> (str):  The path to the file being replaced.



**Returns:**

 - <b>`str`</b>:  Success or error message.

Example call: replace_file(  src_path="config.tmp",  dst_path="config.json" )

---

<a href="../../../../../src/nemantix/stl/local_filesystem/base/py/write_file#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `write_file`

```python
write_file(file_path: str, content: str) → str
```

 Write content to a file. OVERWRITES existing files.  Automatically creates missing parent directories.



**Args:**

         - <b>`file_path`</b> (str):  The path where the file should be written.
         - <b>`content`</b> (str):  The text content to write.



**Returns:**

         - <b>`str`</b>:  Success message.

Example call: write_file(  file_path="notes/todo.txt",  content="1. Buy milk 2. Walk dog" )






---

<a id="emailtoolset"></a>
## <kbd>class</kbd> `EmailToolset`
Initializes the EmailToolset with necessary credentials and server configurations.



**Args:**

 - <b>`email_user`</b> (str, optional):  The email address used for authentication. If None, tries 'EMAIL_USER' env var.
 - <b>`email_password`</b> (str, optional):  The password or app-specific password. If None, tries 'EMAIL_PASSWORD' env var.
 - <b>`smtp_server`</b> (str, optional):  The SMTP server address. Defaults to "smtp.gmail.com".
 - <b>`smtp_port`</b> (int, optional):  The SMTP port (465 for SSL, 587 for STARTTLS). Defaults to 465.
 - <b>`imap_server`</b> (str, optional):  The IMAP server address. Defaults to "imap.gmail.com".
 - <b>`imap_port`</b> (int, optional):  The IMAP port. Defaults to 993.

Example call: EmailToolset(  email_user="user@example.com",  email_password="secretpassword",  smtp_port=587 )

<a href="../../../../../src/nemantix/stl/email_operation/base.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    email_user: Optional[str] = None,
    email_password: Optional[str] = None,
    smtp_server: str = 'smtp.gmail.com',
    smtp_port: int = 465,
    imap_server: str = 'imap.gmail.com',
    imap_port: int = 993
)
```








---

<a href="../../../../../src/nemantix/stl/email_operation/base.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_config`

```python
from_config(config_path: str) → EmailToolset
```

Factory method to create an instance from a JSON configuration file.



**Args:**

 - <b>`config_path`</b> (str):  The file path to the JSON configuration file containing credentials.



**Returns:**

 - <b>`EmailToolset`</b>:  An initialized instance of the EmailToolset class.

Example call: EmailToolset.from_config(  config_path="email_config.json" )

---

<a href="../../../../../src/nemantix/stl/email_operation/base/py/read_emails#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_emails`

```python
read_emails(limit: int = 5, folder: str = 'INBOX') → List[Dict[str, Any]]
```

Reads recent emails via IMAP.



**Args:**

 - <b>`limit`</b> (int, optional):  The maximum number of recent emails to fetch. Defaults to 5.
 - <b>`folder`</b> (str, optional):  The mailbox folder to search in (e.g., "INBOX", "SENT"). Defaults to "INBOX".



**Returns:**

 - <b>`List[Dict[str, Any]]`</b>:  A list of dictionaries, where each dictionary represents an email containing 'from', 'subject', and 'body_preview' keys.

Example call: read_emails(  limit=3,  folder="INBOX" )

---

<a href="../../../../../src/nemantix/stl/email_operation/base/py/send_email#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `send_email`

```python
send_email(recipient: str, subject: str, body: str) → str
```

Sends an email via SMTP (Supports SSL and STARTTLS).



**Args:**

 - <b>`recipient`</b> (str):  The email address of the recipient.
 - <b>`subject`</b> (str):  The subject line of the email.
 - <b>`body`</b> (str):  The plain text content of the email body.



**Returns:**

 - <b>`str`</b>:  A message indicating success or failure of the operation.

Example call: send_email(recipient="john.doe@mail.com",  subject="Test email subject",  body="This is an email." )




---

<a id="requeststoolset"></a>
## <kbd>class</kbd> `RequestsToolset`
A stateless HTTP toolset where authentication is passed explicitly in every request via the 'auth' parameter.

<a href="../../../../../src/nemantix/stl/http_requests/base.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(timeout: int = 10, user_agent: str = 'Agent/1.0')
```

Initialize the HTTP session settings.



**Args:**

 - <b>`timeout`</b> (int):  Global timeout for requests in seconds.
 - <b>`user_agent`</b> (str):  User-Agent header string.




---

<a href="../../../../../src/nemantix/stl/http_requests/base/py/http_delete#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `http_delete`

```python
http_delete(url: str, auth: Optional[Dict[str, str]] = None) → str
```

Performs an HTTP DELETE request to remove a resource.



**Args:**

 - <b>`url`</b> (str):  The URL to request.
 - <b>`auth`</b> (Dict[str, str], optional):  Auth config.



**Returns:**

 - <b>`str`</b>:  The response status and body content.

Example call:
```python
http_delete(
    url="https: //api.example.com/items/42",
    auth={"type":  "bearer", "token": "xyz-987"}
)
```
---

<a href="../../../../../src/nemantix/stl/http_requests/base/py/http_get#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `http_get`

```python
http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    auth: Optional[Dict[str, str]] = None
) → str
```

Performs an HTTP GET request to retrieve data.



**Args:**

 - <b>`url`</b> (str):  The URL to request.
 - <b>`params`</b> (Dict[str, Any], optional):  dictionary of query parameters. Defaults to None.
 - <b>`auth`</b> (Dict[str, str], optional):  Auth config. Examples:
   - `{"type":  "basic", "username": "user", "password": "pass"} `
   - `{"type":  "bearer", "token": "jwt_token"} `
   - `{"type":  "custom", "key": "X-API-KEY", "value": "123"} `



**Returns:**

 - <b>`str`</b>:  The response status, URL, and body content.

Example call:
```python
http_get(
    url="https://api.example.com/users",
    params={"limit": 10},
    auth={"type": "bearer", "token": "abc-123"}
)
```

---

<a href="../../../../../src/nemantix/stl/http_requests/base/py/http_post#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `http_post`

```python
http_post(
    url: str,
    data: Optional[Dict[str, Any]] = None,
    auth: Optional[Dict[str, str]] = None
) → str
```

Performs an HTTP POST request to submit data.



**Args:**

 - <b>`url`</b> (str):  The URL to request.
 - <b>`data`</b> (Dict[str, Any], optional):  The JSON body to send. Defaults to None.
 - <b>`auth`</b> (Dict[str, str], optional):  Auth config (see http_get for examples).



**Returns:**

 - <b>`str`</b>:  The response status, URL, and body content.

Example call:
```python
http_post(
    url="https: //api.example.com/submit",
    data={"name": "Alice", "role": "admin"},
    auth={"type": "custom", "key": "X-API-Key", "value": "secret"}
)
```
---

<a href="../../../../../src/nemantix/stl/http_requests/base/py/http_put#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `http_put`

```python
http_put(
    url: str,
    data: Optional[Dict[str, Any]] = None,
    auth: Optional[Dict[str, str]] = None
) → str
```

Performs an HTTP PUT request to update data.



**Args:**

 - <b>`url`</b> (str):  The URL to request.
 - <b>`data`</b> (Dict[str, Any], optional):  The JSON body to send. Defaults to None.
 - <b>`auth`</b> (Dict[str, str], optional):  Auth config.



**Returns:**

 - <b>`str`</b>:  The response status, URL, and body content.

Example call:
```python
http_put(
    url="https://api.example.com/items/42",
    data={"status": "archived"},
    auth={"type": "basic", "username": "user", "password": "pw"}
)
```



---

<a id="mathsolvertoolset"></a>
## <kbd>class</kbd> `MathSolverToolset`
A toolset for performing advanced symbolic mathematical calculations using the SymPy library.




---

<a href="../../../../../src/nemantix/stl/math_solver/base/py/calculate_derivative#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calculate_derivative`

```python
calculate_derivative(expression: str, variable: str) → str
```

Calculates the symbolic derivative of an expression.



**Args:**

 - <b>`expression`</b> (str):  The function to differentiate.
 - <b>`variable`</b> (str):  The variable with respect to which the derivative is taken.



**Returns:**

 - <b>`str`</b>:  The derivative of the expression.

Example call: calculate_derivative(  expression="sin(x) * x**2",  variable="x" )

---

<a href="../../../../../src/nemantix/stl/math_solver/base/py/calculate_integral#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calculate_integral`

```python
calculate_integral(
    expression: str,
    variable: str,
    lower_limit: Optional[str] = None,
    upper_limit: Optional[str] = None
) → str
```

Calculates the integral of an expression. Performs indefinite integration if limits are omitted.



**Args:**

 - <b>`expression`</b> (str):  The function to integrate.
 - <b>`variable`</b> (str):  The variable of integration.
 - <b>`lower_limit`</b> (str, optional):  The lower bound for definite integration. Defaults to None.
 - <b>`upper_limit`</b> (str, optional):  The upper bound for definite integration. Defaults to None.



**Returns:**

 - <b>`str`</b>:  The result of the integration (symbolic or numeric).

Example call: calculate_integral(  expression="x**2",  variable="x",  lower_limit="0",  upper_limit="3" )

---

<a href="../../../../../src/nemantix/stl/math_solver/base/py/expand_expression#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `expand_expression`

```python
expand_expression(expression: str) → str
```

Expands a factored mathematical expression into a polynomial.



**Args:**

 - <b>`expression`</b> (str):  The expression to expand.



**Returns:**

 - <b>`str`</b>:  The expanded form of the expression.

Example call: expand_expression(  expression="(x + 3) * (x - 2)" )

---

<a href="../../../../../src/nemantix/stl/math_solver/base/py/simplify_expression#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `simplify_expression`

```python
simplify_expression(expression: str) → str
```

Simplifies a mathematical expression algebraically.



**Args:**

 - <b>`expression`</b> (str):  The mathematical expression to simplify.



**Returns:**

 - <b>`str`</b>:  The simplified mathematical expression.

Example call: simplify_expression(  expression="(x + 1)**2 - (x**2 + 2*x + 1)" )

---

<a href="../../../../../src/nemantix/stl/math_solver/base/py/solve_equation#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `solve_equation`

```python
solve_equation(equation: str, variable: str) → str
```

Solves an algebraic equation for a specific variable.



**Args:**

 - <b>`equation`</b> (str):  The equation to solve. If no '=' is present, it assumes the expression equals zero.
 - <b>`variable`</b> (str):  The symbol to solve for.



**Returns:**

 - <b>`str`</b>:  The list of solutions found for the variable.

Example call: solve_equation(  equation="x**2 - 5*x + 6",  variable="x" )




---

<a id="sqlexplorertoolset"></a>
## <kbd>class</kbd> `SqlExplorerToolset`
A Toolset for interacting with a SQL database using SQLAlchemy. Provides tools for schema inspection and query execution.

<a href="../../../../../src/nemantix/stl/sql_explorer/base.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(db_uri: str)
```

Initialize the toolset with a database URI.



**Args:**

 - <b>`db_uri`</b> (str):  SQLAlchemy connection string (e.g., 'sqlite:///example.db').




---

<a href="../../../../../src/nemantix/stl/sql_explorer/base/py/execute_query#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `execute_query`

```python
execute_query(query: str) → str
```

Execute a raw SQL query and return the results. Only SELECT statements should generally be used to ensure safety.



**Args:**

 - <b>`query`</b> (str):  The SQL query string to execute.



**Returns:**

 - <b>`str`</b>:  The query results formatted as a text table, or an error message.

Example call: execute_query(  query="SELECT * FROM users WHERE age > 24" )

---

<a href="../../../../../src/nemantix/stl/sql_explorer/base/py/get_table_schema#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_table_schema`

```python
get_table_schema(table_name: str) → str
```

Get the schema (columns and types) for a specific table.



**Args:**

 - <b>`table_name`</b> (str):  The name of the table to inspect.



**Returns:**

 - <b>`str`</b>:  A formatted string listing columns, types, primary keys, and nullability.

Example call: get_table_schema(  table_name="users" )

---

<a href="../../../../../src/nemantix/stl/sql_explorer/base/py/list_tables#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `list_tables`

```python
list_tables() → str
```

List all accessible table names in the database. Use this to discover what data is available.



**Args:**
  None



**Returns:**

 - <b>`str`</b>:  A comma-separated string of table names, or a message if empty.

Example call: list_tables()




---

<a id="messagingtoolset"></a>
## <kbd>class</kbd> `MessagingToolset`
A toolset for interacting with Telegram Bots. Allows sending messages to specific users via their unique Chat ID.

<a href="../../../../../src/nemantix/stl/messaging/base.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(config_path: Optional[str] = None, bot_token: Optional[str] = None)
```

Initializes the MessagingToolset. It can load the token directly or from a JSON file.



**Args:**

 - <b>`config_path`</b> (str, optional):  The file path to a JSON configuration file containing the 'bot_token'.
 - <b>`bot_token`</b> (str, optional):  The direct Telegram bot token. If config_path is provided, this is ignored.

Example calls: # From JSON: MessagingToolset(config_path="config.json")


---

<a href="../../../../../src/nemantix/stl/messaging/base/py/get_chat_id#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_chat_id`

```python
get_chat_id() → str
```

Check recent messages to find the Chat ID of users who messaged the bot. Use this to find the 'username' needed to send messages to them.



**Returns:**

 - <b>`str`</b>:  A list of users and their Chat IDs.

Example call: get_chat_id()

---

<a href="../../../../../src/nemantix/stl/messaging/base/py/send_message#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `send_message`

```python
send_message(chat_id: str, text: str) → str
```

Send a text message to a specific Telegram account.



**Args:**

 - <b>`chat_id`</b> (str):  The unique numeric ID of the destination account (e.g., "987654321").
 - <b>`text`</b> (str):  The content of the message.



**Returns:**

 - <b>`str`</b>:  Confirmation message.

Example call: send_message(  chat_id="123456789",  text="Hello! This is a message for you." )




---

Next: [Agents](./06%20-%20Agents.md)
