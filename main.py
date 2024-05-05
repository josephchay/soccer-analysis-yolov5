import os
import cv2

from constants import *
from team_assigner import TeamAssigner
from utils import read_video, save_video
from trackers import Tracker


def save_player_imgs(tracks, video_frames, save_dir='outputs/players'):
    # Save the cropped images of the players
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        # crop bbox from frame
        cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # y1:y2, x1:x2

        # save cropped image
        cv2.imwrite(f'{save_dir}/player_{track_id}.jpg', cropped_img)


def main():
    os.makedirs('outputs', exist_ok=True)

    video_frames = read_video(dataset_video_dir + '/08fd33_4.mp4')

    tracker = Tracker('models/yolov5/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/08fd33_4.pkl')

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # save_player_imgs(tracks, video_frames)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, 'outputs/08fd33_4.avi')


if __name__ == "__main__":
    main()
