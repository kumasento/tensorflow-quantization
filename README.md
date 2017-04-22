# tensorflow-quantization

Experiments and tool-chains build upon TensorFlow and its quantization tools.

## Usage (Proposed)

```shell
./main.py \
  --model-name=[MODEL NAME] # The model you would like to evaluate and quantize, model names are specified later
  --quantize-bit=[NUMBER]   # Quantize the model to which bit: 8 or 16
  --model-dir=[MODEL DIR]   # Where to save/restore the model
  --train                   # Train the model or not
```

## Models Implemented

Model files are placed under `models/`.

| Model Name | Source File      | Description                                       |
|------------|------------------|---------------------------------------------------|
| lenet      | `lenet_model.py` | The typical LeNet implementation on MNIST dataset |
