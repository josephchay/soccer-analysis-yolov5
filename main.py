import os

from constants import *
from utils import read_video, save_video


def main():
    os.makedirs('outputs', exist_ok=True)

    video_frames = read_video(dataset_video_dir + '/08fd33_4.mp4')

    save_video(video_frames, 'outputs/08fd33_4.avi')


if __name__ == "__main__":
    main()
