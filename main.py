import os
import cv2

from constants import *
from player_ball_assigner import BallAssigner
from team_assigner import TeamAssigner
from utils import read_video, save_video
from trackers import Tracker


class Main:
    def __init__(self, model_path):
        os.makedirs('outputs', exist_ok=True)

        self.tracker = Tracker(model_path)
        self.video_frames = []
        self.tracks = {}

        self.team_assigner = TeamAssigner()
        self.ball_assigner = BallAssigner()

    def save_player_imgs(self, video_frames, save_dir='outputs/players'):
        # Save the cropped images of the players
        for track_id, player in self.tracks['players'][0].items():
            bbox = player['bbox']
            frame = video_frames[0]

            # crop bbox from frame
            cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # y1:y2, x1:x2

            # save cropped image
            cv2.imwrite(f'{save_dir}/player_{track_id}.jpg', cropped_img)

    def run(self):
        self.video_frames = read_video(DATASET_VIDEO_DIR + '/08fd33_4.mp4')

        self.tracks = self.tracker.get_object_tracks(self.video_frames,
                                                     read_from_stub=True,
                                                     stub_path='stubs/08fd33_4.pkl')

        self.tracks['ball'] = self.tracker.interpolate_ball_positions(self.tracks['ball'])

        # save_player_imgs(tracks, video_frames)

        self.assign_team_colors()
        self.assign_ball_handler()

        output_video_frames = self.tracker.draw_annotations(self.video_frames, self.tracks)

        save_video(output_video_frames, 'outputs/08fd33_4.avi')

    def assign_team_colors(self):
        self.team_assigner.assign_team_color(self.video_frames[0], self.tracks['players'][0])

        for frame_num, player_track in enumerate(self.tracks['players']):
            for player_id, track in player_track.items():
                team = self.team_assigner.get_player_team(self.video_frames[frame_num], track['bbox'], player_id)

                self.tracks['players'][frame_num][player_id]['team'] = team
                self.tracks['players'][frame_num][player_id]['team_color'] = self.team_assigner.team_colors[team]

    def assign_ball_handler(self):
        for frame_num, player_track in enumerate(self.tracks['players']):
            ball_bbox = self.tracks['ball'][frame_num][1]['bbox']
            assigned_ball_handler = self.ball_assigner.assign_to_player(player_track, ball_bbox)

            if assigned_ball_handler != -1:
                self.tracks['players'][frame_num][assigned_ball_handler]['has_ball'] = True


if __name__ == "__main__":
    main = Main(BEST_MODEL_PATH)
    main.run()
