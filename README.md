# Neural Network From Scratch

A handwritten-digit classifier trained on MNIST, built using only NumPy — no PyTorch, no TensorFlow.

**Target accuracy:** ~95% on the MNIST test set.

## Architecture

Input: 28×28 image flattened to 784 numbers  
Output: 10 class scores (digits 0–9)  
Training: forward pass → cross-entropy loss → backprop → gradient descent

## Project structure

```
neural-network-from-scratch/
├── layers.py       # Dense layer (forward + backward)
├── activations.py  # ReLU, softmax
├── losses.py       # Cross-entropy loss
├── network.py      # Chains layers into a full network
├── data_loader.py  # MNIST loading and preprocessing
├── train.py        # Training loop with mini-batches
├── predict.py      # Run inference on new input
├── draw_demo.py    # Tkinter digit-drawing demo
└── saved_models/   # Serialised trained parameters
```

## Milestone checklist

- [ ] Load MNIST and inspect shapes
- [ ] Implement a dense layer with forward pass
- [ ] Implement ReLU
- [ ] Implement numerically stable softmax
- [ ] Implement cross-entropy loss
- [ ] Derive and code backward passes
- [ ] Chain layers into a network
- [ ] Add mini-batch training
- [ ] Track loss and accuracy with plots
- [ ] Save and load trained parameters
- [ ] Add prediction script
- [ ] Add Tkinter digit-drawing demo

## Running

```bash
pip install numpy matplotlib
python train.py
python predict.py
```

## Resources

- 3Blue1Brown — Neural Networks series
- Michael Nielsen, *Neural Networks and Deep Learning*
