# Deep Neural Networks for Magnetic Particle Imaging [UNDER WORK]

My bachelor thesis about denoising of MPI images using neural networks for application in plug-and-play methods.

In this thesis, different neural networks for denoising already reconstructed MPI images will be investigated.

Following image shows the architecture of a convolutional neural network (CNN), which is used in Fast Iterative-Threshold-Shrinkage-Algorithm (FISTA) for denoising MRI images.

<p align="center" style="color:#ff0000">
  <img src="./res/cnn.png">
</p>

### main.py

##### To run the Fista algorithm:
```
$ python main.py
```

##### To run the Fista algorithm with DnCNN: (MUST BE A TRAINED MODEL UNDER Trained_Model DIRECTORY)
```
$ python main.py --nn=True
```

### fista.py

A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. Based on the this [paper](https://epubs.siam.org/doi/pdf/10.1137/080716542)

### model.py

##### To create and train the Model:

```
$ python model.py
```
Trained Models are saved under the directory Trained_Models

Available arguments and their default valuses:

```
$ python model.py --batch_size=128 --epoch=50 --lr=0.0001 --sigma=25
```

##### To test a pre trained model:

```
$ python model.py --test_model=True
```
The test images are saved under images/test/clean

The noisy test images are saved under images/test/noisy 

The the reconstructed images are saved under images/test/output

A CSV report of the results is saved under the directory results
