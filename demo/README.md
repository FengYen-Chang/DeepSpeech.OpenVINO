Python* Demo for Deep Speech
===============================

This is the demo application for Deep Speech algorithm, which make speech to text that are being performed on input speech audio. 

How It Works
------------
The demo expects deep speech models in the Intermediate Representation (IR) format:

It can be your own models, using pre-trained model via model downloader or download and convert pre-trained model by yourself.

1. Download and convert pre-trained model by yourself
    * Download the pre-trained model.
      ```sh
      wget -O - https://github.com/mozilla/DeepSpeech/releases/download/v0.7.4/deepspeech-0.7.4-models.pb
      ```
    * Convert the pre-trained model to Intermediate Representation (IR) by Model Optimizer with the following parameters.
      ```sh
      python3 ./mo_tf.py \
      --input_model=$dl_dir/deepspeech-0.7.4-models.pb \
      --input=input_node,previous_state_h,previous_state_c \
      --input_shape=[1,16,19,26],[1,2048],[1,2048] \
      --output=logits,cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd,cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd_1 \
      --freeze_placeholder_with_value="input_lengths->[16]" \
      --disable_nhwc_to_nchw
      ```
2. Via model downloader
    In the `models.lst` are the list of appropriate models for this demo that can be obtained via `Model downloader`. Please see more information about `Model downloader` [here](../model_downloader/README.md).

Running
-------
Running the application with the `-h` option yields the following usage message:

```
usage: deep_speech_demo.py [-h] -m MODEL -i AUDIO
                           [-d DEVICE]

Options:
  -h, --help            show this help message and exit
  -m  MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i AUDIO, --input AUDIO
                        Required. Required. Path to an audio files.
  -d DEVICE, --device DEVICE
                        Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is
                        acceptable. The demo will look for a suitable plugin for the device specified.
                        Default value is CPU
```

Running Demo

```sh
python3 deep_speech_demo.py -m <path_to_model>/output_graph.xml \
    -i <path_to_audio>/audio.wav
```
Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported. One example wave file can be downloaded from https://github.com/jcsilva/docker-kaldi-gstreamer-server/raw/master/audio/1272-128104-0000.wav.

Demo Output
------------
The application shows the text output for speech audio.

