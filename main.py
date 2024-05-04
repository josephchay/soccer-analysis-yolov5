import os

from constants import *
from utils import read_video, save_video
from trackers import Tracker


def main():
    os.makedirs('outputs', exist_ok=True)

    video_frames = read_video(dataset_video_dir + '/08fd33_4.mp4')

    tracker = Tracker('models/yolov5/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/08fd33_4.pkl')

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, 'outputs/08fd33_4.avi')


if __name__ == "__main__":
    main()
