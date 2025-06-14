from inference import load_model, inference_loop

if __name__ == "__main__":
    model = load_model("model/resnet50_best.pt")
    inference_loop(model)