# bad-apple-files

(This is supposed to be a bad apple archive, but it's just a bad apple implementation for the task in Python Discord CJ10 for now)

Formally speaking, this repository contains an implementation of the "Bad Apple!!" animation for the Python Discord Code Jam 10 (CJ10). The `extract_frames.py` script is designed to extract frames from a video, preprocess them, and optionally save them as video fragments for further use.

## Usage

To use the `extract_frames.py` script, run it from the command line with the following arguments:

```bash
python extract_frames.py --file_path "path/to/video.mp4" --save_path "path/to/save/fragments/" --image_path "path/to/image.png" --ordering_path "path/to/ordering.txt" --frame_range start_frame end_frame --frames_per_save frames_per_save
```

### Arguments

- `--file_path`: Path to the input video file.
- `--save_path`: Path where video fragments will be saved.
- `--image_path`: Path to the image used for unscrambling.
- `--ordering_path`: Path to the ordering of the tiles.
- `--frame_range`: Range of frames to extract, specified as two integers `start_frame` and `end_frame`.
- `--frames_per_save`: Number of frames to include in each video fragment.

### Example

For example, if you want to extract frames from the video "bad_apple.mp4" in the "bad_apple" directory, process frames within the range of 0 to 300, and save video fragments in the "video_fragments" directory:

```bash
python extract_frames.py --file_path "bad_apple/bad_apple.mp4" --save_path "video_fragments/" --frame_range 0 300 --image_path "image/great_wave_scrambled.png" --ordering_path "image/great_wave_order.txt" 
```
(create the save directory first)  

This command would preprocess frames within the specified range and save them as video fragments, each containing the number of frames specified by `frames_per_save`. Additionally, you can provide optional arguments like `image_path` and `ordering_path` if you have specific image and ordering files to use for preprocessing.  

Finally, run `concatenate_video.py` to concatenate the fragments. if your video fragments aren't saved in `video_fragments/`, change `save_path` in line 30.
## Requirements

See `requirements.txt`