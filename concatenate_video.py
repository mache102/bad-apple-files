import os 
import subprocess

ZF = 4

def concatenate_video_fragments(save_path: str, clean_files: bool = False) -> None:
    """
    Concatenate video fragments using FFmpeg.

    :param save_path: Path where video fragments are saved.
    """
    fragment_files = [f"{str(i).zfill(ZF)}.mp4" for i in range(len(os.listdir(save_path)))]
    concat_file = os.path.join(save_path, "concat_list.txt")

    with open(concat_file, "w") as f:
        for fragment_file in fragment_files:
            f.write(f"file '{fragment_file}'\n")
            

    output_file = os.path.join(save_path, "final_video.mp4")
    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", output_file])

    # Clean up intermediate files
    if clean_files:
        for fragment_file in fragment_files:
            os.remove(fragment_file)
        os.remove(concat_file)

def main():
    save_path = "video_fragments/"
    concatenate_video_fragments(save_path)

if __name__ == "__main__":
    main()