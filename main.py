import os
import cv2
import numpy as np

from constants import *
from player_ball_assigner import BallAssigner
from team_assigner import TeamAssigner
from utils import read_video, save_video
from trackers import Tracker
from camera_movement import CameraMovement


class Main:
    def __init__(self, model_path):
        os.makedirs('outputs', exist_ok=True)

        self.tracker = Tracker(model_path)
        self.video_frames = []
        self.tracks = {}

        self.team_assigner = TeamAssigner()
        self.ball_assigner = BallAssigner()

        self.team_ball_control = []

    def save_player_imgs(self, save_dir='outputs/players'):
        os.makedirs(save_dir, exist_ok=True)

        # Save the cropped images of the players
        for track_id, player in self.tracks['players'][0].items():
            bbox = player['bbox']
            frame = self.video_frames[0]

            # crop bbox from frame
            cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # y1:y2, x1:x2

            # save cropped image
            cv2.imwrite(f'{save_dir}/player_{track_id}.jpg', cropped_img)
            print(f'Saved player_{track_id}.jpg')

    def run(self):
        os.makedirs('outputs', exist_ok=True)

        self.video_frames = read_video(DATASET_VIDEO_DIR + '/08fd33_4.mp4')

        self.tracks = self.tracker.get_object_tracks(self.video_frames,
                                                     read_from_stub=True,
                                                     stub_path='stubs/08fd33_4.pkl')

        self.tracker.add_position_to_tracks(self.tracks)

        # camera movement
        camera_movement = CameraMovement(self.video_frames[0])
        camera_movement_per_frame = camera_movement.get_camera_movement(self.video_frames,
                                                                        read_from_stub=True,
                                                                        stub_path='stubs/camera_movement.pkl')

        camera_movement.add_adjust_positions_to_tracks(self.tracks, camera_movement_per_frame)

        self.tracks['ball'] = self.tracker.interpolate_ball_positions(self.tracks['ball'])

        # self.save_player_imgs()

        self.assign_team_colors()
        self.assign_ball_handles()

        output_video_frames = self.tracker.draw_annotations(self.video_frames, self.tracks, self.team_ball_control)

        output_video_frames = camera_movement.draw_camera_movement(output_video_frames, camera_movement_per_frame)

        save_video(output_video_frames, 'outputs/08fd33_4.avi')

    def assign_team_colors(self):
        self.team_assigner.assign_team_color(self.video_frames[0], self.tracks['players'][0])

        for frame_num, player_track in enumerate(self.tracks['players']):
            for player_id, track in player_track.items():
                team = self.team_assigner.get_player_team(self.video_frames[frame_num], track['bbox'], player_id)

                self.tracks['players'][frame_num][player_id]['team'] = team
                self.tracks['players'][frame_num][player_id]['team_color'] = self.team_assigner.team_colors[team]

    def assign_ball_handles(self):
        for frame_num, player_track in enumerate(self.tracks['players']):
            ball_bbox = self.tracks['ball'][frame_num][1]['bbox']
            assigned_ball_handler = self.ball_assigner.assign_to_player(player_track, ball_bbox)

            if assigned_ball_handler != -1:
                self.tracks['players'][frame_num][assigned_ball_handler]['has_ball'] = True
                self.team_ball_control.append(self.tracks['players'][frame_num][assigned_ball_handler]['team'])
            else:
                self.team_ball_control.append(self.team_ball_control[-1])

        self.team_ball_control = np.array(self.team_ball_control)


if __name__ == "__main__":
    main = Main(BEST_MODEL_PATH)
    main.run()
