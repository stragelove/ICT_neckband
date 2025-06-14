import torch
import torch.nn.functional as F
import preprocessor as pp
import sounddevice as sd

# sd.default.device = (0)

MODEL = "model/resnet50_best.pt"
CLASS_NAME = ["alarm", "alarm", "alarm", "non", "alarm"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
DURATION = 1
NUM_SAMPLE = int(SAMPLE_RATE * DURATION)

# 모델 로드
def load_model(model_path):
    try:
        model = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.to(DEVICE)
        model.eval()
        return model
    
    except FileNotFoundError as e:
        print(f"모델 파일을 찾을 수 없습니다: {e}")
        return None

# 예측
def predict(model, logmel, device=DEVICE):
    """
    logmel: numpy.ndarray 형태 (1, C, H, W)
    반환: (예측 인덱스, 확률 배열)
    """
    x = torch.from_numpy(logmel).to(device)

    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(prob.argmax())

    return idx, prob

# 실시간 마이크 에측
def inference_loop(model):
    """
    마이크로부터 실시간으로 입력 받아 예측을 계속 출력합니다.
    Ctrl-C 로 종료.
    """
    print("실시간 마이크 입력을 시작합니다.")

    try:
        while True:
            waveform = sd.rec(frames=NUM_SAMPLE,
                            samplerate=SAMPLE_RATE,
                            channels=1,
                            dtype="float32"
                            ).transpose(1, 0)
            sd.wait()

            logmel = pp.logmel(source=waveform, num_samples=NUM_SAMPLE)
            idx, prob = predict(model, logmel)

            print(f"{CLASS_NAME[idx]} || {prob[idx] * 100:.2f}%")

    except KeyboardInterrupt as e:
        print("\n실시간 마이크 입력을 종료합니다.\n", e)
