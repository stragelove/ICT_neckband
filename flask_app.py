import json
import sounddevice as sd
from flask import Flask, Response, stream_with_context, render_template

import preprocessor as pp
from inference import load_model, predict, MODEL, SAMPLE_RATE, CLASS_NAME

app = Flask(__name__)
model = load_model(MODEL)

CHUNK_DURATION = 1
CHUNK_SAMPLES  = int(SAMPLE_RATE * CHUNK_DURATION)

@app.route("/stream")
def stream():
    """
    서버 마이크에서 1초 녹음 → logmel → 예측 → JSON 반환
    """
    def event_stream():
        while True:
            # 녹음
            recording = sd.rec(CHUNK_SAMPLES, SAMPLE_RATE, 1, "float32").transpose(1, 0)
            sd.wait()

            # 전처리 & 추론
            logmel = pp.logmel(source=recording, num_samples=CHUNK_SAMPLES)
            idx, prob = predict(model, logmel)

            # JSON 반환
            data = json.dumps({
                "클래스": CLASS_NAME[idx],
                "확률": float(prob[idx])
            }, ensure_ascii=False)
            
            yield f"data: {data}\n\n"
            
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
