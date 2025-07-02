from ultralytics import YOLO

# 모델 불러오기 (pretrained segment 모델)
model = YOLO('yolo11n-seg.pt')  # n: nano, s: small, m: medium, l: large

# 학습하기
model.train(
    data='datasets_combined/data.yaml',  # 데이터셋 yaml 경로
    epochs=100,
    imgsz=640,
    batch=8,
    workers=4,
    device=0,  # 0번 GPU 사용
    project='runs/segment',
    name='add_data',
    optimizer='SGD',  # 기본은 SGD, 필요하면 Adam도 가능
    lr0=0.01,  # 초기 learning rate
    pretrained=True  # pretrained backbone 사용
)
