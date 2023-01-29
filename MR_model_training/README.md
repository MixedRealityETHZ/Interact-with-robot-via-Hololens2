# Hand gesture recognition model training
In this folder, we provide the training code for the hand gesture recognition model.

## Dataset
We collect the training data using MRTK and save the hand joint data to txt files within the `data` folder. The dataset is parsed in `dataset.py`.

## Training

We use Multi-layer perceptron (MLP) to train the MR model. The network architecture is defined in `model.py`. Please check `train.py` for more training details.

After runing the `train.py`, the PyTorch model will be converted to onnx format and can be later imported to Unity and inferenced by Barracuda.