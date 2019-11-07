Model Downloader for Deep Speech
-------------------------------------------------------------------------
This is a sample for download and convert the pre-trained model by model downloader.

How It Works
-----------------------
1. Please move the folder `deep-speech-v0.5.0` to `$OPENVINO_ROOT/deployment_tools/open_model_zoo/models/public`

    ```sh
    mv $ROOT_OF_REPO/model_downloader/deep-speech-v0.5.0 $OPENVINO_ROOT/deployment_tools/open_model_zoo/models/public/deep-speech-v0.5.0
    ```
2. Run `downlaoder.py` to download the pre-trained model

    ```sh
    python downloader.py --name deep-speech-v0.5.0
    ```
3. Run `converter.py`to convert the pre-trained model to Intermediate Representation (IR) format

    ```sh
    python converter.py --name deep-speech-v0.5.0
    ```
