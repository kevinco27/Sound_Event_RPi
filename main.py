import caffe2.python.onnx.backend
import numpy as np
import onnx
import argparse
import pyaudio
import time
from collections import deque
from utils import audio_feature
from utils.led_vis import Event_Light


model_path = './model/epoch_56.onnx'

# params for audio feature extraction (mel-spectrogram)
parser = argparse.ArgumentParser(description='Real Time Sound Event Detection')
parser.add_argument('--dn',  default='CRW_baby_cry', type=str, help='dataset name')
parser.add_argument('--sr',  default=16000, type=int, help='[fea_ext] sample rate')
parser.add_argument('--ws',  default=2000,  type=int, help='[fea_ext] windows size')
parser.add_argument('--wws',  default=2048,  type=int, help='[fea_ext] windows size')
parser.add_argument('--hs',  default=497,   type=int, help='[fea_ext] hop size')
parser.add_argument('--mel', default=128,   type=int, help='[fea_ext] mel bands')
parser.add_argument('--msc', default=1,     type=int, help='[fea_ext] top duration of audio clip')
parser.add_argument('--et',  default=10000, type=int, help='[fea_ext] spect manti')
parser.add_argument('--ch', default=1, type=int, help='channels of microphone')
args = parser.parse_args()

# load model
print "load model: ", model_path
model = onnx.load(model_path)
prepared_backend = caffe2.python.onnx.backend.prepare(model)  # using caffe2 as backend

# audio streaming and sound event detecting
p = pyaudio.PyAudio()
stream = p.open(
        format=pyaudio.paFloat32,
        channels=args.ch,
        rate=args.sr,
        input=True,
        frames_per_buffer=args.ws
        )

print "start streaming and event detecting"
result_buf = deque(np.zeros(5), maxlen=5)
event_light = Event_Light()
while(True):
    try:
        frames = []
        # read streaming data
        for i in range(0, int(args.sr/args.ws * args.msc)):
            data = stream.read(args.ws, exception_on_overflow=False)
            data = np.fromstring(np.array(data), np.float32)
            frames.extend(data)
        # turn audio to melspectrogram and send to NN model
        fea = audio_feature.get_mel(frames, args).astype("float32")
        output = prepared_backend.run(fea)
        output = np.argmax(output)
        print output
        result_buf.append(output)
        if np.sum(result_buf) >= 2:
            event_light.event_On()
        else:
            event_light.event_Off()

    except KeyboardInterrupt:
        event_light.light_Off()
        break
print "Terminating"

stream.stop_stream()
time.sleep(0.5)
stream.close()
p.terminate()

