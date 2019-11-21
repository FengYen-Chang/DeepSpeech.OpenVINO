Accuracy Checker for Deep Speech
-------------------------------------------------------------------------
This is a sample for enable the accuracy checker to execute the Deep Speech.

Prepare the dataset libriSpeech
-----------------------
Suggest using this [script](https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/librispeech.py) which from this [repo](https://github.com/SeanNaren/deepspeech.pytorch) download the dataset and convert the `.flac` into `.wav` with 16kHz. Or download the dataset from [Librispeech](http://www.openslr.org/12) and convert the `.flac` into `.wav` with 16kHz by yourself.

Enable it
-------------------------
In this repo, we provide several files, please following below step to enable it.

### Annotation converter
Please move [libri_speech.py](./annotation_converters/libri_speech.py) from this repo into direction `$OPEN_MODEL_ZOO_DIR/tools/accuracy_checker/accuracy_checker/annotation_converters/`, and then following below step to register the `libri_speech` converter.
  * Register paser into `__init__.py` which under direction `${OPENVINO_INSTALL_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checher/accuracy_checher/annotation_converters/`.

      1. Import `libri_speech`

          ```py
          from .libri_speech import LibriSpeechFormatConverter
          ```

      2. Add class `LibriSpeechFormatConverter` into list `__all__`

          ```py
          __all__ = [

              ...

              'ActionRecognitionConverter',
              'LibriSpeechFormatConverter'
          ]
          ```

### Data reader
Please check [data_reader.py](./data_readers/data_reader.py) and move class [`AudioReader`](https://github.com/FengYen-Chang/DeepSpeechOpenVINO/blob/master/accuracy_checker/data_readers/data_reader.py#L168) into file `$OPEN_MODEL_ZOO_DIR/tools/accuracy_checker/accuracy_checker/data_readers/data_reader.py`, and then following below step to register the `aduio` reader.
  * Register paser into `__init__.py` which under direction `${OPENVINO_INSTALL_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checher/accuracy_checher/data_readers/`.

      1. Import `AudioReader`
    
          ```py
          from .data_reader import (

              ...

              create_reader,
              AudioReader
          )
          ```
    
      2. Add class `AudioReader` into list `__all__`

          ```py
          __all__ = [

              ...

              'create_reader',
              'AudioReader'
          ]
          ```
      
### Input feeder
Please follow below step to patch `input_feeder.py` which under direction `$OPEN_MODEL_ZOO_DIR/tools/accuracy_checker/accuracy_checker/launcher/`.
  * Add new input type `HIDDEN_STATE` into `input_feeder.py` 

    ```py
    if input_['type'] == 'CONST_INPUT':
        if isinstance(value, list):
            value = np.array(value)
        constant_inputs[name] = value
    elif input_['type'] == 'HIDDEN_STATE':
        constant_inputs[name] = np.zeros(tuple(input_['shape']))
    else:
        config_non_constant_inputs.append(name)
    ```
    >Note: Please refer to line [148](https://github.com/FengYen-Chang/DeepSpeechOpenVINO/blob/master/accuracy_checker/launcher/input_feeder.py#L148)
   
### Pre-processor
Please move [audio_preprocessors.py](./preprocessor/audio_preprocessors.py) from this repo into direction `$OPEN_MODEL_ZOO_DIR/tools/accuracy_checker/accuracy_checker/preprocessor/`, and then following below step to register the pre-processors.
  * Register  into `__init__.py` which under direction `${OPENVINO_INSTALL_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checher/accuracy_checher/preprocessor/`.

      1. From `audio_preprocessors` import each function which just implemented. 
    
         ```py
         from .audio_preprocessors import (
             AudioNormalizer,
             AudioSpectrogram,
             MfccFeature,
             CreateOverlapWindows,
             PrepareAudioPackage
         )
         ```
    
    2. Add classes into list `__all__`

         ```py
         __all__ = [

             ...

             'PadWithEOS',

             'AudioNormalizer',
             'AudioSpectrogram',
             'MfccFeature',
             'CreateOverlapWindows',
             'PrepareAudioPackage'
         ]
         ```

### Launcher

Please follow below step to patch `dlsdk_launcher.py` which under direction `$OPEN_MODEL_ZOO_DIR/tools/accuracy_checker/accuracy_checker/launcher/`.
  
  * Add flag `run_audio` and `audio_hidden_state` into function `parameters` which under class `DLSDKLauncher`.

     ```py
     @classmethod
     def parameters(cls):
         parameters = super().parameters()
         parameters.update({
             'model': PathField(description="Path to model."),
             'weights': PathField(description="Path to model."),

             ...

             '_vpu_log_level': StringField(
                 optional=True, choices=VPU_LOG_LEVELS, description="VPU LOG level: {}".format(', '.join(VPU_LOG_LEVELS))
             ),
             'run_audio': BoolField(
                 optional=True, default=False,
                 description="The specific flag for speech recognition to run the predict function(deep speech only)."
             ),
             'audio_hidden_state': ListField(optional=True, description="audio hidden state(deep speech only).")
         })

         return parameters
     ```
     >Note: Please refer to line [209](https://github.com/FengYen-Chang/DeepSpeechOpenVINO/blob/master/accuracy_checker/launcher/dlsdk_launcher.py#L209)

 * Read the value from flag `run_audio` and `audio_hidden_state` ans save it as parameter in the function `__init__` which under class `DLSDKLauncher`

     ```py
     def __init__(self, config_entry, delayed_model_loading=False):
         super().__init__(config_entry)

         ...

         self.reload_network = not delayed_model_loading

         self.run_audio = get_parameter_value_from_config(self.config, DLSDKLauncher.parameters(), 'run_audio')
         self.audio_hidden_state = get_parameter_value_from_config(self.config, DLSDKLauncher.parameters(), 'audio_hidden_state')
     ```
     >Note: Please refer to line [247](https://github.com/FengYen-Chang/DeepSpeechOpenVINO/blob/master/accuracy_checker/launcher/dlsdk_launcher.py#L247)

 * Patch the function `predict` which under class `DLSDKLauncher` to add the inference process for Deep Speech and using flag `run_audio` to distinguish it from original process.

     ```py
     def predict(self, inputs, metadata=None, **kwargs):
         results = []
         for infer_inputs in inputs:
             if self.run_audio:
                 audio_ftrs = infer_inputs[self.config['_list_inputs'][0]]
                 hidden_state = []
                 __res = None
                 output_node = None

                 hidden_state.append(infer_inputs[self.config['_list_hidden_states'][0]])
                 hidden_state.append(infer_inputs[self.config['_list_hidden_states'][1]])

                 for _, __audio_ftr in enumerate(audio_ftrs):
                     network_inputs_data = {self.config['_list_inputs'][0] : [__audio_ftr],
                                             self.config['_list_hidden_states'][0] : hidden_state[0],
                                             self.config['_list_hidden_states'][1] : hidden_state[1]}

                     result = self.exec_network.infer(network_inputs_data)

                     if not output_node:
                         output_nodes = list(result.keys())
                         output_nodes.remove(self.audio_hidden_state[0])
                         output_nodes.remove(self.audio_hidden_state[1])
                         output_node = output_nodes[0]
                         __res = np.empty([0, 1, result[output_node].shape[-1]])

                     hidden_state[0] = result[self.audio_hidden_state[0]]
                     hidden_state[1] = result[self.audio_hidden_state[1]]

                     __res = np.concatenate((__res, result[output_node]))

                 results.append({output_node :__res})

             else:
                 if self._do_reshape:
                     input_shapes = {layer_name: data.shape for layer_name, data in infer_inputs.items()}
                     self._reshape_input(input_shapes)

                 result = self.exec_network.infer(infer_inputs)
                 results.append(result)

         if metadata is not None:
             for meta_ in metadata:
                 meta_['input_shape'] = self.inputs_info_for_meta()

         self._do_reshape = False

         return results
     ```
     >Note: Please refer to line [278](https://github.com/FengYen-Chang/DeepSpeechOpenVINO/blob/master/accuracy_checker/launcher/dlsdk_launcher.py#L278)

 * Patch the function `_align_data_shape` which under class `DLSDKLauncher` to align data shape for audio data.

     ```py
     def _align_data_shape(self, data, input_blob):
         input_shape = self.network.inputs[input_blob].shape

         if self.run_audio:
             if data.shape[1:] != input_shape[1:]:
                 warning_message = 'data shape {} is not equal model input shape {}. '.format(
                     data.shape[1:], input_shape[1:]
                     )
             return data

         data_batch_size = data.shape[0]

         ...

     ```
     >Note: Please refer to line [537](https://github.com/FengYen-Chang/DeepSpeechOpenVINO/blob/master/accuracy_checker/launcher/dlsdk_launcher.py#L537)

 * Patch the function `fit_to_input` which under class `DLSDKLauncher` to reshape the input.

     ```py
     def fit_to_input(self, data, layer_name, layout):
         def data_to_blob(layer_shape, data):
             data_shape = np.shape(data)
             if len(layer_shape) == 4:
                 if len(data_shape) == 5:
                     data = data[0]
                 if self.run_audio:
                     return data
                 else:
                     return np.transpose(data, layout)

             if len(layer_shape) == 2 and len(data_shape) == 1:

             ...

     ```
     >Note: Please refer to line [688](https://github.com/FengYen-Chang/DeepSpeechOpenVINO/blob/master/accuracy_checker/launcher/dlsdk_launcher.py#L688)

### Adapter
Please follow below step to patch `text_detection.py` which under direction `$OPEN_MODEL_ZOO_DIR/tools/accuracy_checker/accuracy_checker/adapters/`.

 * Add flag `output_node` into function `parameters` which under class `BeamSearchDecoder` in file `${OPENVINO_INSTALL_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checher/accuracy_checher/adapters/text_detection.py`. Here set the flag as an optional since not all topology need to assign.

     ```py
     def parameters(cls):
         parameters = super().parameters()
         parameters.update({

             ...

             'softmaxed_probabilities': BoolField(
                 optional=True, default=False, description="Indicator that model uses softmax for output layer "
             ),
             'output_node': StringField(
                 optional=True, description="for assign a specific output node (default will using the node from launcher.output_blob)"
             )
         })
         return parameters
     ```
     >Note: Please refer to line [712](https://github.com/FengYen-Chang/DeepSpeechOpenVINO/blob/master/accuracy_checker/adapters/text_detection.py#L712)

 * Parse the flag `output_node` in function `configure` which under class `BeamSearchDecoder`.

     ```py
     def configure(self):
         if not self.label_map:
             raise ConfigError('Beam Search Decoder requires dataset label map for correct decoding.')

         ...

         self.softmaxed_probabilities = self.get_value_from_config('softmaxed_probabilities')
         self.output_node = self.get_value_from_config('output_node')
     ```
     >Note: Please refer to line [728](https://github.com/FengYen-Chang/DeepSpeechOpenVINO/blob/master/accuracy_checker/adapters/text_detection.py#L728)

 * Check the flag `output_node` in function `process` which under class `BeamSearchDecoder`, and using it if assigned.

     ```py
     def process(self, raw, identifiers=None, frame_meta=None):
         raw_output = self._extract_predictions(raw, frame_meta)
         if (self.output_node) :
             output = raw_output[self.output_node]
         else:
             output = raw_output[self.output_blob]

         output = np.swapaxes(output, 0, 1)
     ```
     >Note: Please refer to line [732](https://github.com/FengYen-Chang/DeepSpeechOpenVINO/blob/master/accuracy_checker/adapters/text_detection.py#L732)

### Metrics

Please move [speech_recognition.py](./metrics/speech_recognition.py) and [word_error_meter.py](./metrics/word_error_meter.py) from this repo into direction `$OPEN_MODEL_ZOO_DIR/tools/accuracy_checker/accuracy_checker/metrics/`, and then following below step to register the metric.
  * Register  into `__init__.py` which under direction `${OPENVINO_INSTALL_DIR}/deployment_tools/open_model_zoo/tools/accuracy_checher/accuracy_checher/metrics/`.

       1. Import `SpeechRecognitionAccuracy`
         
           ```py

           ...

           from .character_recognition import CharacterRecognitionAccuracy
           from .speech_recognition import SpeechRecognitionAccuracy

           ...

           ```
         
       2. Add class `SpeechRecognitionAccuracy` into list `__all__`

           ```py
           __all__ = [

               ...

               'ExactMatchScore',
               'SpeechRecognitionAccuracy'
           ]
           ```




