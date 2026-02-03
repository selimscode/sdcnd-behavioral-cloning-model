# Behavioral Cloning: Self-Driving Car

Train a CNN to steer a car autonomously by learning from human driving behavior.

![Demo](media/sdc_model_track1_gif.gif)

## Approach

1. **Collect training data** - Drive manually in simulator, capture camera frames + steering angles
2. **Balance the dataset** - Fix steering bias using left/right camera images
3. **Augment aggressively** - Crop, resize, blur, flip, randomize brightness
4. **Train NVIDIA-style CNN** - 5 conv layers → 4 dense layers → steering angle output

## Architecture

Modified [NVIDIA self-driving architecture](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) optimized for faster training:

| Layer | Output Shape | Details |
|-------|--------------|---------|
| Lambda | 64×64×3 | Normalize to [-0.5, 0.5] |
| Conv2D | 30×30×24 | 5×5 filter, stride 2, ReLU |
| Conv2D | 13×13×36 | 5×5 filter, stride 2, ReLU |
| Conv2D | 5×5×48 | 5×5 filter, stride 2, ReLU |
| Conv2D | 3×3×64 | 3×3 filter, stride 2, ReLU |
| Conv2D | 1×1×64 | 3×3 filter, stride 2, ReLU |
| Dense | 80 → 40 → 20 → 10 → 1 | Dropout 50% between layers |

**Total params**: 140,829

## Data Pipeline

| Step | Purpose |
|------|---------|
| Crop 160×320 → 75×320 | Remove sky and hood |
| Resize → 64×64 | Reduce memory footprint |
| BGR → RGB | Match inference colorspace |
| Gaussian blur | Reduce noise, improve generalization |
| Random brightness | Simulate lighting variation |
| Horizontal flip | Double data, balance left/right turns |

## Key Insight: Steering Balance

Raw data had heavy center bias (driver stays centered). This caused the model to fail on curves.

**Solution**: Use left/right camera images with steering correction to balance the distribution:
- Before: mean=-0.011, heavily center-weighted
- After: mean=0.002, balanced across left/center/right

## Results

The model completes the track smoothly, staying centered through curves and straightaways.

**Training**: 8,000 images (sampled from 32,000 balanced set), 3 epochs, batch size 128

## Files

- `model.py` - Training script with augmentation pipeline
- `model.h5` - Trained Keras model weights
- `drive.py` - Connects model to Udacity simulator
