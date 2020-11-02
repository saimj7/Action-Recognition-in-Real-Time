# Video-Classification-in-Real-Time

Video classification using VGG16 as a feature extractor and seasoning with RNN. Dataset used is UCF101 (Cricket bowling and batting classes).

> A simple RNN is used to better classify temporal frame sequences from videos.

Cricket batting           |  Cricket batting/bowling
:-------------------------:|:-------------------------:
![bat](mylib/misc/bat.gif?raw=true "bat")  |  ![bowl](mylib/misc/bowl.gif?raw=true "bpwl") 

--- 

> Some of the use cases would be monitoring anomalies, suspicious human actions, alerting the staff/authorities.

## Table of Contents
* [Background Theory](#background-theory)
* [Running Inference](#running-inference)
* [Pipeline](#pipeline)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
* [References](#references)
* [Next Steps](#next-steps)

## Background Theory
**Feature extraction:**
- Pretrained VGG16 is used as a feature extractor after fine tuning/unfreezing its 4 top layers.
- A simple classifier is then connected to VGG16 and trained to identify if the frame belongs to class1 or 2. 
- Then the top classifier is disconnected and only dense layer with 1024 output size is used to obtain the sparse representations of each frame. 
- **Data to lstm format:** For each video frame, the sparse representations are stacked into a tensor of size (NUM_FRAMES, LOOK_BACK, 1024). 

<div align="center">
<img src="https://github.com/saimj7/Video-Classification-in-Real-Time/blob/master/mylib/misc/model.jpg" width=570>
<p>- Model architecture -</p>
</div>

---

**RNN:**
- A standard LSTM is used. Note that you need GPU/CUDA support if you would like to run CUDnnLSTM layers in the model. 
- Finally, the LSTM network is trained to distinguish between your desired class1 and 2 videos.

## Running Inference
- Install all the required Python dependencies:
```
pip install -r requirements.txt
```
- To run inference either on a test video file or on webcam: 
```
python run.py 
```
- Note that the inference is set on the test video file by default. 
- To change it, simply set ``` FROM_WEBCAM = True ``` in the config. options at mylib/Config.py
- Trained model weights (for this example) can be downloaded from [**here**](https://drive.google.com/file/d/1mGm9jnZhelskbSzYAWWQoNcGa3mz95OL/view?usp=drivesdk). Make sure you extract them into the folder 'weights'.
- The class probabilities and inference time per frames is also displayed:
```
[INFO] Frame acc. predictions: 0.91895014
Frame inference in 0.0030 seconds
```
- You can also chose to send prediction accuracies over the mail if desired. Follow the instructions in mylib>Mailer.py (to setup the sender mail).
- Enter the receiver mail in the config. options at mylib/Config.py

<div align="center">
<img src="https://github.com/saimj7/Video-Classification-in-Real-Time/blob/master/mylib/misc/alert.jpg" width=500>
<p>- Predictions alert -</p>
</div>

- In case of severe false positivies, make sure to optimize the threshold and positive_frames parameters to further narrow down the predictions. Please refer config.
```
Threshold = 0.50
if pred >= Threshold:
```

```
if total_frames > 5:
   print('[INFO] Sending mail...')
```


## Pipeline

### Preprocessing:
- Some image processing is required before training on your own data! 
- In 'Preprocessing.ipynb' file, the frames from each video classes are extracted and sorted into respective folders.
- Note that the frames are resized to 224x224 dimensions (which is VGG16 input layer size).
- The dataset can be downloaded from [**here**](https://www.crcv.ucf.edu/data/UCF101.php).

### Training:
- 'Train.ipynb', as the name implies trains your model.
- Training is visualized with the help of TensorBoard. Use the command:
```
tensorboard --logdir data/_training_logs/rnn
```
<div align="center">
<img src="https://github.com/saimj7/Video-Classification-in-Real-Time/blob/master/mylib/misc/train.jpg" width=470>
<p>- Training accuracy -</p>
</div>

- Make sure to review the parameters in config. options at mylib/Config.py
- You will come across the parameters in Train.ipynb, they must be same during the training and inference.
- If you would like to change them, simply do so in the training file and also in config. options.

## References

***Main:***
- VGG16 paper: https://arxiv.org/pdf/1409.1556.pdf
- UCF101 Action Recognition Data Set: https://www.crcv.ucf.edu/data/UCF101.php

***Optional:***
- TensorBoard: https://www.tensorflow.org/tensorboard

## Next steps
- Investigate and benchmark different RNN architectures for better classifying the temporal sequences.

<p>&nbsp;</p>

---

## Thanks for the read & have fun!

> To get started/contribute quickly (optional) ...

- **Option 1**
    - 🍴 Fork this repo and pull request!

- **Option 2**
    - 👯 Clone this repo:
    ```
    $ git clone https://github.com/saimj7/Action-Recognition-in-Real-Time.git
    ```

- **Roll it!**

---

saimj7/ 06-09-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.
