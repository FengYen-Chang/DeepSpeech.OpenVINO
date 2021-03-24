Python* Demo for Deep Speech
===============================

This is the demo application for Deep Speech algorithm, which make speech to text that are being performed on input speech audio. 

How It Works
------------
The demo expects deep speech models in the Intermediate Representation (IR) format:

It can be your own models, using pre-trained model via model downloader or download and convert pre-trained model by yourself.

* **For English Model**
   
   This is the sample for convert the deep speech version `0.7.4` English model. For further version, please check the [Speech Recognition Demo](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/python_demos/speech_recognition_demo/README.md) page on OMZ.

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

* **For Chinese Model**

   This is the sample for convert the deep speech version `0.9.3` Chinese model. To convert this model to Intermediate Representation (IR) format, we need to get a script from [OMZ](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/mozilla-deepspeech-0.8.2/pbmm_to_pb.py) to convert the `.pbmm` to `.pb` format.
   
   1. Download and convert pre-trained model by yourself
       * Download the pre-trained model.
         ```sh
         wget -O - https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models-zh-CN.pbmm
         ```
       * Convert the model from `.pbmm` to `,pb`
         ```sh
         python3 $repo_dir/utils/pbmm_to_pb.py $model_dir/deepspeech-0.9.3-models-zh-CN.pbmm $dl_dir/deepspeech-0.9.3-models-zh-CN.pb
         ```
         > This file, `pbmm_to_pb.py`, is based on [OMZ](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/mozilla-deepspeech-0.8.2/pbmm_to_pb.py). It could convert the `.pbmm` to `.pb`. However, if use the orignal `pbmm_to_pb.py` to convert the model to `.pb`, the converted `.pb` will occur the ascii code error during convert the model to IR. Therefore, I replace the constant node, `metadata_alpahbet`, by another node, `metadata_language`, to avoid the error. As this constant is no needed in final graph, this workround is fine. And I will find out the best way to solve the problem in future. 
       * Convert the pre-trained model to Intermediate Representation (IR) by Model Optimizer with the following parameters.
         ```sh
         python3 ./mo_tf.py \
         --input_model=$dl_dir/deepspeech-0.9.3-models-zh-CN.pb \
         --input=input_node,previous_state_h,previous_state_c \
         --input_shape=[1,16,19,26],[1,2048],[1,2048] \
         --output=logits,cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd,cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd_1 \
         --freeze_placeholder_with_value="input_lengths->[16]" \
         --disable_nhwc_to_nchw
         ```

Running
-------
Running the application with the `-h` option yields the following usage message:

```
usage: deep_speech_demo.py [-h] -m MODEL -i INPUT -l LANGUAGE [-d DEVICE]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to an audio file.
  -l LANGUAGE, --language LANGUAGE
                        Required. Define the trained model is for CN or ENG.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. The
                        sample will look for a suitable plugin for device
                        specified. Default value is CPU

```

Running Demo

* **For English Model**
  ```sh
  python3 deep_speech_demo.py -m <path_to_model>/output_eng_graph.xml \
      -i <path_to_audio>/audio.wav -l ENG
  ```
  Only 16-bit, 16 kHz, mono-channel WAVE audio files are supported. One example wave file can be downloaded from https://github.com/jcsilva/docker-kaldi-gstreamer-server/raw/master/audio/1272-128104-0000.wav.

* **For Chinese Model**
  ```sh
  python3 deep_speech_demo.py -m <path_to_model>/output_cn_graph.xml \
      -i <path_to_audio>/audio.wav -l CN
  ```
Demo Output
------------
The application shows the text output for speech audio.

