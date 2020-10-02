import numpy as np
import cv2, re, os, keras, warnings, time
from keras.models import Model
import keras.layers as L
from keras.optimizers import RMSprop
from keras import models
from keras import optimizers
from keras.applications import VGG16
from mylib.Mailer import Mailer
from mylib import Config
warnings.filterwarnings('ignore')

CAM_CONSTANT = 0

# CNN VGG model
class FeatExtractor:
    def __init__(self, SIZE):
        self.size = config.SIZE
        self.vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(self.size[0], self.size[1], 3))
        for layer in self.vgg_conv.layers[:-4]:
            layer.trainable = False

        # Create the model
        def build_feat_extractor():
            model = models.Sequential()

            # Add the vgg convolutional base model
            model.add(self.vgg_conv)

            # Add new layers
            model.add(L.Flatten())
            model.add(L.Dense(1024, activation='relu'))
            model.add(L.Dropout(0.2))
            model.add(L.Dense(256, activation='relu'))
            model.add(L.Dense(2, activation='softmax'))
            return model

        self.model = build_feat_extractor()
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])

        self.model.load_weights('weights/Feature_Extractor.h5')

        inp = self.model.input
        out = self.model.layers[-4].output
        self.model = Model(inputs=[inp], outputs=[out])

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])

    def get_feats(self, frames):
        image_data = np.zeros((len(frames), config.VGG16_OUT))
        for index, image in enumerate(frames):
            vect = self.model.predict(image.reshape(1, self.size[0], self.size[1], 3))
            image_data[index, :] = vect

        image_data = image_data.reshape(1, len(frames), config.VGG16_OUT)
        return image_data

# RNN model
class RnnModel:

    def __init__(self, NUM_FEATURES, LOOK_BACK):
        self.num_features = NUM_FEATURES
        self.look_back = config.LOOK_BACK
        def build_model():
            inp = L.Input(shape=(self.look_back, self.num_features))
            x = L.LSTM(64, return_sequences=True)(inp)
            x = L.Dropout(0.2)(x)
            x = L.LSTM(16)(x)

            out = L.Dense(2, activation='softmax')(x)
            model = Model(inputs=[inp], outputs=[out])
            model.compile(loss='categorical_crossentropy',
                          optimizer=RMSprop(lr=1e-4),
                          metrics=['acc'])
            return model

        self.model = build_model()
        self.model.load_weights('weights/RNN.h5')

    def predict(self, frame_data):
        pred = self.model.predict(frame_data)
        return pred[0][1]

def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

#===============================================================================
# Initiate the main function
if __name__ == '__main__':
    if not config.FROM_WEBCAM:
        # Enter your desired test video path
        cap = cv2.VideoCapture('tests/v_CricketShot_g22_c01.avi')
    else:
        # From webcam
        cap = cv2.VideoCapture(CAM_CONSTANT, cv2.CAP_DSHOW)
    cnt = 0
    frames = []
    fe = FeatExtractor(config.SIZE)
    rnn = RnnModel(config.VGG16_OUT, config.LOOK_BACK)
    total_frames = 0
    detect_certainty = []
    neg_certainty = []
    while (cap.isOpened()):
        # Capture frame-by-frame
        cnt+=1
        ret, full = cap.read()
        frame = cv2.resize(full, config.SIZE)
        if cnt % config.TAKE_FRAME == 0:
            frames.append(frame)
            pred = 0
            if len(frames) == config.LOOK_BACK:
                # Get features
                feats = fe.get_feats(frames)
                frames.pop(0)
                initial = time.time()
                pred = rnn.predict(feats)
                final = time.time() - initial
                print("")
                # Check predictions per frame (either 0 or 1)
                print('[INFO] Frame acc. predictions:', pred)
                # Check inference time per frame
                print('Frame inference in %.4f seconds' % (final))

            if ret == True:
                # Display the resulting frame
                # Optimize the threshold (avg. prediction score for class labels) if desired
                # 1 for class1 and 0 for class2. Please refer config.
                if pred >= config.Threshold:
                    __draw_label(full, 'Bowl', (20, 20), (255, 255, 255))
                    total_frames += 1
                    detect_certainty.append(pred)
                else:
                    neg_certainty.append(pred)
                    if config.ALERT:
                        # Adjust the total_frames (avg. score to send the mail). Refer config.
                        if total_frames > config.positive_frames:
                            print('[INFO] Sending mail...')
                            neg = np.mean(neg_certainty)
                            pos = np.mean(detect_certainty)
                            time1 = total_frames * config.TAKE_FRAME / 30
                            Mailer().send(config.MAIL, total_frames, time1, pos, neg)
                            print('[INFO] Mail sent')
                        detect_certainty = []
                        total_frames = 0
                    __draw_label(full, 'Bat', (20, 20), (255, 255, 255))
                cv2.imshow('Test_Window', full)

                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

    # When everything done, release the video capture object
    cap.release()
