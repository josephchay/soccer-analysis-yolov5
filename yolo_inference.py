from ultralytics import YOLO
import sys
import gdown

from constants import *
from utils.filesystem import *

# Check if the datasets already exists
if not os.path.exists(dataset_video_dir):
    os.makedirs('assets', exist_ok=True)

    down_destination = dataset_video_dir + '.zip'

    gdown.download(dataset_video_url, down_destination, quiet=False)
    print("Downloaded dataset video zip file")

    try:
        extract_zip(down_destination, 'assets')
    except Exception as e:
        name = os.path.basename(sys.argv[0])
        print("At file: " + name[:name.rfind('.')] + ", unable to extract " + down_destination + " to " + 'assets')

test_video_path = dataset_video_dir + '/08fd33_4.mp4'

# model = YOLO('original.pt')
model = YOLO('models/yolov5/best.pt')

results = model.predict(
    source=test_video_path,
    save=True,
    project=".",
    name="runs/detect/predict"
)

print(results[0])
for box in results[0].boxes:
    print(box)
