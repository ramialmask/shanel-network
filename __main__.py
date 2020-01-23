import argparse
import training.training as training
import prediction.prediction as prediction
import counting.counting as counting
import json
from utilities.util import create_meta_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Inference on or Test a model")
    parser.add_argument("mode", metavar="M", type=str, default="training", help="Usage mode: (training, prediction, test_model(folder))")
    parser.add_argument("-use_gpu", metavar="G", type=int, default=-1, help="Manually assign a gpu number")
    args = parser.parse_args()
    mode = args.mode
    use_gpu = args.use_gpu

    settings = create_meta_dict(".", mode)

    # load settings
    # settings = {}
    # path = "./meta_data.json"
    # with open(path) as file:
    #     settings = json.loads(file.read())


    #TODO set gpu

    if mode == "train":
        print("Check out TensorBoard to track the training progress:  https://localhost:6006")
        training.testfold_training(settings)
    elif mode == "predict":
        prediction.prediction(settings)
    elif mode == "count":
        counting.counting(settings)
