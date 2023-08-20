import argparse 
import os 
import subprocess

ZF = 4

"""
python3 concatenate_video.py --save_path crop_video_fragments/ --audio_path bad_apple/bad_apple.mp3
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate video fragments using FFmpeg.")

    parser.add_argument("--save_path", type=str, default="video_fragments/", 
                        help="Path where video fragments are saved.")
    parser.add_argument("--clean_files", action="store_true", 
                        help="Whether to clean up intermediate files.")
    parser.add_argument("--audio_path", type=str, default=None, 
                        help="Path to the audio file to add to the video.")

    args = parser.parse_args()
    return args

def concatenate_video_fragments(args: argparse.Namespace) -> None:
    """
    Concatenate video fragments using FFmpeg.

    :param save_path: Path where video fragments are saved.
    """
    fragment_files = [f"{str(i).zfill(ZF)}.mp4" for i in range(len(os.listdir(args.save_path)))]
    concat_file = os.path.join(args.save_path, "concat_list.txt")

    with open(concat_file, "w") as f:
        for fragment_file in fragment_files:
            f.write(f"file '{fragment_file}'\n")
            

    output_file = os.path.join(args.save_path, "final_video.mp4")

    if args.audio_path is not None:
        subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file, "-i", args.audio_path, "-c", "copy", output_file])
    else:
        subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", output_file])

    # Clean up intermediate files
    if args.clean_files:
        for fragment_file in fragment_files:
            os.remove(fragment_file)
        os.remove(concat_file)

def main(args):
    concatenate_video_fragments(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)