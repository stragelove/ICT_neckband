import torch
import torch.nn.functional as F
import Preprocessor as pp
import sounddevice as sd

sd.default.device = (0)

MODEL = "model/resnet50_best.pt"
CLASS_NAME = ["alarm", "alarm", "alarm", "non", "alarm"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
DURATION = 1
NUM_SAMPLE = int(SAMPLE_RATE * DURATION)

def predict(model, logmel, device=DEVICE):
    logmel = torch.from_numpy(logmel)
    logmel = logmel.to(device)

    with torch.no_grad():
        outputs = model(logmel)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        idx = probabilities.argmax()

    return idx, probabilities

def inference(model):

    try:
        model = torch.load(model, map_location=DEVICE, weights_only=False)
        model.to(DEVICE)
        model.eval()

    except FileNotFoundError as e:
        print(f"모델 파일을 찾을 수 없습니다: {e}")
        return
        

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
            idx, probabilities = predict(model, logmel)

            print(f"{CLASS_NAME[idx]} || {probabilities[idx] * 100:.2f}%")

    except KeyboardInterrupt as e:
        print("\n실시간 마이크 입력을 종료합니다.\n", e)

if __name__ == "__main__":
    model = MODEL
    inference(model)