from collections import defaultdict

import cv2  #영상처리 라이브러리
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  # 학습시킨 모델 가져오기
video = cv2.VideoCapture("raw.mp4")  # 영상 가져오기
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("runs/detect/track.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
history = defaultdict(list)  # 지나온 중심점을 저장

for _ in range(round(fps * 5)):  # 5초 동안 영상처리
    ok, frame = video.read()
    if not ok:  # 영상이 5초 전에 끝나면 멈춘다
        break
    result = model.track(frame, persist=True, tracker="botsort.yaml")[0]
    image = result.plot()  # 상자, 추적 번호, 부류 이름, 신뢰도가 포함된 프레임
    if result.boxes.is_track:  # 추적 번호가 붙은 물체가 있을 때만
        for box in result.boxes:
            track_id = int(box.id)
            x, y, w, h = box.xywh[0]
            history[track_id].append((x, y))
            points = np.hstack(history[track_id]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [points], isClosed=False, color=(255, 0, 0), thickness=4)  # 지나온 중심점을 선으로 이어 시각화
    writer.write(image)
video.release()
writer.release()