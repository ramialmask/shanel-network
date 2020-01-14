import torch
import argparse
import training.training as training
import prediction.prediction as prediction
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Inference on or Test a model")
    parser.add_argument("mode", metavar="M", type=str, default="training", help="Usage mode: (training, prediction, test_model(folder))")
    parser.add_argument("-use_gpu", metavar="G", type=int, default=-1, help="Manually assign a gpu number")
    args = parser.parse_args()
    mode = args.mode
    use_gpu = args.use_gpu

    # load settings
    settings = {}
    path = "./meta_data.json"
    with open(path) as file:
        settings = json.loads(file.read())

    #TODO print stats

    # maybe print link to Tensorboard?
    print("Check out TensorBoard:  https://localhost:6006")

    #TODO set gpu
    torch.cuda.init()
    torch.cuda.set_device(0)
    
    if mode == "train":
        training.testfold_training(settings)
    elif mode == "predict":
        prediction.prediction(settings)
    #TODO train, test, predict
