Accuracy Checker for Deep Speech
-------------------------------------------------------------------------
This is a sample for enable the accuracy checker to execute the Deep Speech.

Prepare the dataset libriSpeech
-----------------------
Suggest using this [script](https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/librispeech.py) which from this [repo](https://github.com/SeanNaren/deepspeech.pytorch) download the dataset and convert the `.flac` into `.wav` with 16kHz. Or download the dataset from [Librispeech](http://www.openslr.org/12) and convert the `.flac` into `.wav` with 16kHz by yourself.

Make it work
-------------------------
In this repo, we provide several files, please following below step to enable it.

* Annotation converter
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

* Data reader
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
      










