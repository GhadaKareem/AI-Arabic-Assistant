from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import os
import playsound
from glob import glob
import numpy as np
import pandas as pd
import random
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import sounddevice as sd
import soundfile as sf
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from gtts import gTTS

class DatasetGenerator():
    def __init__(self, label_set,
                 sample_rate=16000):

        self.label_set = label_set
        self.sample_rate = sample_rate

    # Covert string to numerical classes
    def text_to_labels(self, text):
        return self.label_set.index(text)

    # Reverse translation of numerical classes back to characters
    def labels_to_text(self, labels):
        #print("labels :"+str(labels))
        return self.label_set[labels]

    def load_data(self, DIR):
        # Get all paths inside DIR that ends with wav
        data = []
        for i in range(len(DIR)):
            for filename in glob(os.path.join(DIR[i], '*.wav')):
                wav_files = filename
                show = wav_files.split('\\')

                label = show[2]
                name = show[3]
                if label in self.label_set:
                    label_id = self.text_to_labels(label)
                    fle = os.path.join(DIR[i], name)
                    sample = (label, label_id, name, fle)
                    data.append(sample)
        #print(str(data))
        # Data Frames with samples' labels and paths
        df = pd.DataFrame(data, columns=['label', 'label_id', 'user_id', 'wav_file'])

        self.df = df

        return self.df

    def apply_train_test_split(self, test_size, random_state):

        self.df_train, self.df_test = train_test_split(self.df,
                                                       test_size=test_size,
                                                       random_state=random_state)

    def apply_train_val_split(self, val_size, random_state):

        self.df_train, self.df_val = train_test_split(self.df_train,
                                                      test_size=val_size,
                                                      random_state=random_state)

    def read_wav_file(self, x):
        # Read wavfile using scipy wavfile.read
        _, wav = wavfile.read(x)
        # Normalize
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max

        return wav

    def process_wav_file(self, x, threshold_freq=5500, eps=1e-10):
        # Read wav file to array

        wav = self.read_wav_file(x)
        # Sample rate
        L = self.sample_rate
        # If longer then randomly truncate
        if len(wav) > L:
            i = np.random.randint(0, len(wav) - L)
            wav = wav[i:(i + L)]
            # If shorter then randomly add silence
        elif len(wav) < L:
            rem_len = L - len(wav)
            silence_part = np.random.randint(-100, 100, 16000).astype(np.float32) / np.iinfo(np.int16).max
            j = np.random.randint(0, rem_len)
            silence_part_left = silence_part[0:j]
            silence_part_right = silence_part[j:rem_len]
            wav = np.concatenate([silence_part_left, wav, silence_part_right])
        # Create spectrogram using discrete FFT (change basis to frequencies)
        freqs, times, spec = stft(wav, L, nperseg=400, noverlap=240, nfft=512, padded=False, boundary=None)
        # Cut high frequencies
        if threshold_freq is not None:
            spec = spec[freqs <= threshold_freq, :]
            freqs = freqs[freqs <= threshold_freq]
        # Log spectrogram
        amp = np.log(np.abs(spec) + eps)

        return np.expand_dims(amp, axis=2)

    def generator(self, batch_size, mode):
        while True:
            # Depending on mode select DataFrame with paths
            if mode == 'train':
                df = self.df_train
                ids = random.sample(range(df.shape[0]), df.shape[0])
            elif mode == 'val':
                df = self.df_val
                ids = list(range(df.shape[0]))
            elif mode == 'test':
                df = self.df_test
                #print("df test:  p " + str(df))
                ids = list(range(df.shape[0]))
            elif mode == 'prediction':
                X_batch = []
                X_batch.append(self.process_wav_file(r"C:\Users\Ghada Gamal\Desktop\Record.wav"))
                X_batch = np.array(X_batch)

                yield X_batch
            else:
                raise ValueError('The mode should be either train, val or test.')

            # Create batches (for training data the batches are randomly permuted)
            if mode != 'prediction':

                for start in range(0, len(ids), batch_size):
                    X_batch = []
                    if mode != 'test':
                        y_batch = []
                    end = min(start + batch_size, len(ids))
                    i_batch = ids[start:end]
                    for i in i_batch:
                        X_batch.append(self.process_wav_file(df.wav_file.values[i]))
                        if mode != 'test':
                            y_batch.append(df.label_id.values[i])
                    X_batch = np.array(X_batch)

                    if mode != 'test':
                        y_batch = to_categorical(y_batch, num_classes=len(self.label_set))
                        yield (X_batch, y_batch)
                    else:
                        yield X_batch

DIR = ["D:\Speech_GP\سؤال الدكتور","D:\Speech_GP\رعايه نفسيه","D:\Speech_GP\التقرير الطبي","D:\Speech_GP\قياس معدل ضربات القلب و الضغط","D:\Speech_GP\قياس درجة الحرارة" ]
#DIR = ["سؤال الدكتور","رعايه نفسيه","قراءه التقرير الطبي","قياس الضغط و معدل ضربات القلب","قياس درجه الحراره"]
INPUT_SHAPE = (177,98,1)
BATCH = 32
EPOCHS = 15

with open("demofile.txt", "r", encoding="utf-8") as f:
    LABELS = [line.rstrip() for line in f]

print("list of labels:" + str(LABELS))
NUM_CLASSES = len(LABELS)
chatbot = ChatBot("RobotNurse")
trainer = ListTrainer(chatbot)
trainer.train([
    "التقرير الطبي",
    "سأقرأ التقرير الطبي",
])
trainer.train([
    "التقرير الطبي",
    "سأقرأ التقرير الطبي",
])
trainer.train([
    "رعايه نفسيه",
    "إهدأ لا داعي للقلق",
])
trainer.train([
    "سؤال الدكتور",
    "سأتصل بالدكتور حالا",
])

trainer.train([
    "سؤال الدكتور",
    "سأتصل بالطبيب حالا",
])
trainer.train([
    "رعايه نفسيه",
    "إهدأ ستكون بخير",
])
trainer.train([
    "قياس معدل ضربات القلب و الضغط",
    "حاضر",
])
trainer.train([
    "قياس درجة الحرارة",
    "سأقيس درجة الحرارة",
])


dsGen = DatasetGenerator(label_set=LABELS)
# Load DataFrame with paths/labels for training and validation data
# and paths for testing data
df = dsGen.load_data(DIR)
dsGen.apply_train_test_split(test_size=0.3, random_state=2018)
dsGen.apply_train_val_split(val_size=0.2, random_state=2018)


def deep_cnn(features_shape, num_classes, act='relu'):
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x

    # Block 1
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block1_conv', input_shape=features_shape)(o)
    o = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block1_pool')(o)
    o = BatchNormalization(name='block1_norm')(o)

    # Block 2
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block2_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(o)
    o = BatchNormalization(name='block2_norm')(o)

    # Block 3
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block3_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(o)
    o = BatchNormalization(name='block3_norm')(o)

    # Flatten
    o = Flatten(name='flatten')(o)

    # Dense layer
    o = Dense(64, activation=act, name='dense')(o)
    o = BatchNormalization(name='dense_norm')(o)
    o = Dropout(0.2, name='dropout')(o)

    # Predictions
    o = Dense(num_classes, activation='softmax', name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=o).summary()

    return Model(inputs=x, outputs=o)

def train ():
    model = deep_cnn(INPUT_SHAPE, NUM_CLASSES)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

    mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks = EarlyStopping(monitor='val_acc', patience=4, verbose=1, mode='max')
    history = model.fit_generator(generator=dsGen.generator(BATCH, mode='train'),
                                  steps_per_epoch=int(np.ceil(len(dsGen.df_train) / BATCH)),
                                  epochs=EPOCHS,
                                  callbacks=[callbacks, mc],
                                  verbose=1,
                                  validation_data=dsGen.generator(BATCH, mode='val'),
                                  validation_steps=int(np.ceil(len(dsGen.df_val) / BATCH)))
    #test
    y_pred_proba = model.predict_generator(dsGen.generator(BATCH, mode='test'),
                                           int(np.ceil(len(dsGen.df_test) / BATCH)), verbose=1)

    y_pred = np.argmax(y_pred_proba, axis=1)

    pre = []
    for cell in y_pred.flatten():
        #print("cell :"+ str(cell))
        pre = dsGen.labels_to_text(cell)
        #print("you mean: " + str(pre))

    y_true = dsGen.df_test['label_id'].values

    acc_score = accuracy_score(y_true, y_pred)

    print(acc_score)

def predict():
    model = deep_cnn(INPUT_SHAPE, NUM_CLASSES)
    model = load_model('best_model.hdf5')
    while True :
        User_input = input("Are you ready y or n: ").lower()
        if User_input == 'n':
            break
        elif User_input == 'y':
            samplerate = 16000
            duration = 3  # seconds
            filename = r'C:\Users\Ghada Gamal\Desktop\Record.wav'
            print("Speak! ")
            mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
            print("Stop!! ")
            sd.wait()
            sf.write(filename, mydata, samplerate)
            y_pred_proba = model.predict_generator(dsGen.generator(BATCH, mode='prediction'),int(np.ceil((len(dsGen.df_test)+len(dsGen.df_train)) / BATCH)),verbose=1)
            y_pred = np.argmax(y_pred_proba, axis=1)
            pre = []
            lst=list(y_pred.flatten())
           # print("count 0 " + str(lst.count(0)))
           # print("count 1 " + str(lst.count(1)))
           # print("count 2 " + str(lst.count(2)))
           # print("count 3 " + str(lst.count(3)))
           # print("count 4 " + str(lst.count(4)))
            max_index = 0
            for i in range(len(LABELS)):
                if lst.count(max_index) < lst.count(i):
                    max_index = i
            pre = dsGen.labels_to_text(max_index)

            #print("you mean: " + str(pre))

            Seq2Seq(str(pre))
        else:
            print('Wrong Input!')
            continue

def Seq2Seq(prediction_word):
    try:
        bot_input = chatbot.get_response(prediction_word)
        print(bot_input)
        Text2Speech(str(bot_input))
    except(KeyboardInterrupt, EOFError, SystemExit):
        print("Error Happened")

def Text2Speech(sentence):
    tts = gTTS(sentence ,lang="ar")
    tts.save("Text2Speech.mp3")
    playsound.playsound("Text2Speech.mp3")
    os.remove("Text2Speech.mp3")

def main():
    User_input = input("Enter train or Prediction: ").lower()
    if User_input == 't':
        train()
    elif User_input == 'p':
        predict()
    else:
        print('Wrong Input!')
if __name__ == '__main__':
    main()

