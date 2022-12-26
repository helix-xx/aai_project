# aai_project
### Features of voice:
Basically, we can use spectrum analysis. The simplest is use FFT on voice file. 

### Data Process:
First, we turn the flac file into wav file. Cut or paste the file to a fixed length, 12800. Then do the FFT in dataset. So when load data, FFT and training process will work at same time.

### Model selection:
We notice that voice recognition is also a process of recognize feature. So at the very beginning, we apply Resnet.

### Model training and pred
We train our model on CSE cluster. we use Adam optimizer and Cross Entropy loss function. We split 20% of train data as test dataset.
After 50 epochs, we could get about 90% accuracy on test data set.
