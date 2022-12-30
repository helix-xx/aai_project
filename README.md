# aai_project
A simple CNN model for speaker recognition.

### Features of voice:
Basically, we can use spectrum analysis.   
The simplest is use FFT on voice file. For voice data, structure is 1D. We could implement FFT to get 1D input or STFT to get 2D input.  
Another way is extract voice feature by MFCC, FBank or LogFBank. MFCC usually use for traditional GMM-HMM model. For CNN, FBank has less imformation loss.


### Data Process:
First, we turn the flac file into wav file. Cut or paste the file to a fixed length. Then do the FFT in dataset. So when load data, FFT and training process will work at same time. In this way, we could get 90% accuracy on no-noise dataset.
For improve, we augment the dataset by speed up 1.05x and slow down 0.95x. we combine them and cut to 4s fragment. Then randomly add noise and do the Mel filter to get FBank features. Before send to model, do normalize to each Mel feature.

### Model structrue:
We notice that voice recognition is also a process of recognize feature. We apply Resnet1D and Resnet2D for 1D and 2D input. They have same structure. Basic block has three conv layer and one residual block.  
At the beginning, we use conv layer to transform 1 inchannel to 32 output channels, and a maxpool layer for downsample. Then we apply 5 groups of block, each groups has `[3, 4, 6, 3, 3]` blocks. At the top, we use avgool and fullyconnect layer to trainsform output size to 250*1, which is the number of classes of the speaker. Add drop to prevent overfit.

### Model training and pred
We train our model on CSE cluster. we use Adam optimizer and CrossEntropy loss function. We split 20% of train data as test dataset. After 50 epochs, we could get about 90% accuracy on test data set.
Step1: Use `flac2way.py` transfrom data to `.wav`file.  
Step2: Use `generate_data.py` to augment data.
Step3: Check all the data is available to use.
Step4: Use `train.py` to train the model, it will save model in specified path.
Step5: Use `pred.py` to predict the result, it will save result in specified path.

Parameters can be change in `data_process.py`, `path` need be specified in each file.
