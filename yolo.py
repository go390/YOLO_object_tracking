import io
import sys


class print_log:
    #화면 출력을 그대로 보여 주면서 같은 내용을 로그에도 남긴다.
    def __init__(self, stream):
        self.stream = stream      # 원래 화면 출력
        self.log = io.StringIO()  # 결과 폴더가 정해지기 전까지 출력을 모아 둘 메모리

    def write(self, text):
        self.log.write(text)
        return self.stream.write(text)

    def start_file(self, trainer):
        early = self.log.getvalue()
        self.log = open(trainer.save_dir / "train.log", "w", encoding="utf-8", buffering=1)
        self.log.write(early)

    def __getattr__(self, name):
        return getattr(self.stream, name)

if __name__ == "__main__":
    prl = print_log(sys.stdout)
    sys.stdout = sys.stderr = prl

    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    model.add_callback("on_pretrain_routine_start", prl.start_file)
    results = model.train(data="YOLO.v1i.yolov8/data.yaml", epochs=100, imgsz=640)
