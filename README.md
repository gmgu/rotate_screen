# rotate_screen

Vidoes in Youtube are sometimes rotated in 90 degrees or 270 degrees.
This project detects the angle of captured screen images by a CNN (convolutional neural netowork)

## Trained CNN

Trained CNN is sotred in trained_model/cnn.pth

## Example

Run main.py will show a running example of this proejct.
```
python3 main.py
```

## Accuracy
Current accuracy is as follows (using 256_cnn.pth)
Accuracy of the network: 81.37335526315789 %
Accuracy of 0: 78.27160493827161 %
Accuracy of 90: 80.8877928483354 %
Accuracy of 270: 84.95684340320592 %

### History
CenterCrop 714 (256_cnn_c.pth)
- Accuracy of the network: 76.27467105263158 %
- Accuracy of 0: 75.30864197530865 %
- Accuracy of 90: 73.61282367447596 %
- Accuracy of 270: 79.90135635018495 %

