import argparse 
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os 
import warnings

from PIL import Image
from typing import Callable, Tuple, List
from tqdm import tqdm

ZF = 4

def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video and preprocess them.")

    parser.add_argument("--file_path", type=str, help="Path to the input video file.")
    parser.add_argument("--save_path", type=str, default=None, help="Path where video fragments will be saved.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the image to unscramble.")
    parser.add_argument("--ordering_path", type=str, default=None, help="Path to the ordering of the tiles.")

    parser.add_argument("--frame_range", type=int, nargs=2, help="Range of frames to extract.")
    parser.add_argument("--frames_per_save", type=int, default=30, help="Number of frames to include in each video fragment.")

    args = parser.parse_args()
    return args


def extract_frames(file_path: str, frame_range: List[int], func: Callable[[np.ndarray, float, str], None], frame_size: Tuple[int, int] = None, save_path: str = None, frames_per_save: int = 30, **func_kwargs) -> None:
    """
    Extract frames from a video, preprocess them, and pass to the provided function.

    :param file_path: Path to the input video file.
    :param func: Function to process the frames. Should accept arguments (np.ndarray, float, str).
    :param frame_size: Tuple (width, height) to resize frames. If None, no resizing is done.
    :param save_path: Path where video fragments will be saved.
    :param frames_per_save: Number of frames to include in each video fragment.
    :param func_args: Additional positional arguments to pass to the provided function.
    :param func_kwargs: Additional keyword arguments to pass to the provided function.
    """
    success = True
    vid = cv2.VideoCapture(file_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_range[0] = max(0, frame_range[0])
    frame_range[1] = min(frame_range[1], max_frames)

    fig = plt.figure()
    ims = []

    if frame_range[0] > 0:
        fragment_count = np.ceil(frame_range[0] / frames_per_save).astype(int)
    else:
        fragment_count = 0
    frames_in_fragment = 0
    skipped_frames = 0

    for i in tqdm(range(frame_range[1])):
        success, img = vid.read()
        if success:
            if skipped_frames <= frame_range[0]:
                skipped_frames += 1
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.array(img, dtype=np.uint8)
            img = np.swapaxes(img, 0, 1)

            if frame_size is not None:
                img = cv2.resize(img, frame_size)

            frame = func(img, fps, **func_kwargs)
            im = plt.imshow(frame, animated=True)
            ims.append([im])

            frames_in_fragment += 1
            if frames_in_fragment >= frames_per_save:    

                ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=1000)
                fragment_save_path = os.path.join(save_path, f"{str(fragment_count).zfill(ZF)}.mp4")
                ani.save(fragment_save_path, writer='ffmpeg')
                fragment_count += 1
                frames_in_fragment = 0
                fig = plt.figure()
                ims = []

        else:
            warnings.warn(f"Frame {i} not found")

    if ims:
        ani = animation.ArtistAnimation(fig, ims, interval=1000 // fps, blit=True, repeat_delay=1000)
        fragment_save_path = os.path.join(save_path, f"{str(fragment_count).zfill(ZF)}.mp4")
        ani.save(fragment_save_path, writer='ffmpeg')



def bad_apple_cj_qual(frame: np.ndarray, fps: float, image: Image.Image, ordering: List[int], tile_size: Tuple[int, int]):

    _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

    # downscale the tile size and image size by res
    res = 4
    tile_size = (tile_size[0] // res, tile_size[1] // res)
    image = image.resize((image.size[0] // res, image.size[1] // res))

    tile_width, tile_height = tile_size
    image_width, image_height = image.size

    rows = image_height // tile_height
    cols = image_width // tile_width

    image_modded = image.copy()

    for row in range(rows):
        for col in range(cols):

            # unscramble only if the frame at this point is white
            if frame[col, row] != 0:
                continue 

            new_tile_pos = np.copy(ordering[row * cols + col])
            
            new_row = new_tile_pos // cols
            new_col = new_tile_pos % cols    

            new_tile = image.crop((new_col * tile_width, new_row * tile_height, 
                                  (new_col + 1) * tile_width, (new_row + 1) * tile_height))

            image_modded.paste(new_tile, (col * tile_width, row * tile_height))
    
    return np.array(image_modded)

def main(args):
    # args.file_path = "bad_apple/bad_apple.mp4"
    # args.save_path = "video_fragments/"
    # args.image_path = "image/great_wave_scrambled.png"
    # args.ordering_path = "image/great_wave_order.txt"

    # args.frame_range = [600, 900]
    # args.frames_per_save = 30
    # the image is (1104,1600) with tile size (16,16)
    # frame_size is then (1104//16,1600//16) = (69,100)
    frame_size = (69, 100)
    tile_size = (16, 16)

    image = Image.open(args.image_path)

    with open(args.ordering_path, 'r') as f:
        ordering = [int(x) for x in f.read().strip().splitlines()]

    extract_frames(file_path=args.file_path, frame_range=args.frame_range, 
                   func=bad_apple_cj_qual, frame_size=frame_size, 
                   image=image, ordering=ordering, tile_size=tile_size,
                   save_path=args.save_path, frames_per_save=args.frames_per_save)

if __name__ == "__main__":
    args = parse_args()
    main(args)