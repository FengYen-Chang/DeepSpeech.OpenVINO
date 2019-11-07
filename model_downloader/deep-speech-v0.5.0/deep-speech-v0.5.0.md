# deep-speech-v0.5.0

## Use Case and High-Level Description

DeepSpeech is an open source Speech-To-Text engine, using a model trained by machine learning techniques based on [Baidu's Deep Speech research paper](https://arxiv.org/abs/1412.5567). The model was trained on the LibriSpeech clean test corpus for 16-bit, 16 kHz, mono-channel WAVE audio files. Project DeepSpeech uses Google's [TensorFlow](https://www.tensorflow.org/) to make the implementation easier. 

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Speech recognition                        |
| GFlops                          | 416.97                                    |
| MParams                         | 47.225                                    |
| Source framework                | TensorFlow\*                              |

## Performance

## Input

### Original Model

1. Audio, name: `input_samples`, a WAVE file.
2. Audio Feature, name: `input_node`, shape: [1x16x19x26], format: [BxSxCxI],
   where:

    - B - batch size
    - S - number of steps
    - C - number of context
    - I - number of input
    
3. Hidden state, name: `previous_state_h/read`, shape: [1x2048], format: [BxF],
   where: 
    
    - B - batch size
    - F - number of features in the hidden state `h`
    
    > Note: The initial value of hidden state `h` is zero.

4. Hidden state, name: `previous_state_c/read`, shape: [1x2048], format: [BxF], 
   where: 
   
    - B - batch size
    - F - number of features in the hidden state `c`
    
    > Note: The initial value of hidden state `c` is zero.

### Converted Model

1. Audio Feature, name: `input_node`, shape: [1x16x19x26], format: [BxSxCxI],
   where:

    - B - batch size
    - S - number of steps
    - C - number of context
    - I - number of input

    > Note: Needs to acquire the mfcc feature for audio.
    
2. Hidden state, name: `previous_state_h/read/placeholder_port_0`, shape: [1x2048], format: [BxF],
   where: 
    
    - B - batch size
    - F - number of features in the hidden state `h`
    
    > Note: The initial value of hidden state `h` is zero.

3. Hidden state, name: `previous_state_c/read/placeholder_port_0`, shape: [1x2048], format: [BxF], 
   where: 
   
    - B - batch size
    - F - number of features in the hidden state `c`
    
    > Note: The initial value of hidden state `c` is zero.

## Output

### Original Model

1. Text, name: `Softmax`, shape: [Nx1xM], format: [IxBxL]
   where: 
    
    - I - number of iteration
    - B - batch size
    - L - number of label
    
    > Note: The number of iteration is depend on the feature length. To make the model can execute with different length of input feature, the model will be iterated since the input shape was fixed.
    
2. Hidden state `h`, name: `lstm_fused_cell/BlockLSTM:6`, shape: [1x2048], format: [BxF],
   where: 
    
    - B - batch size
    - F - number of features in the hidden state `h`
    
    > Note: The previous hidden state will be the initial state for next iteration.  
    
3. Hidden state `c`, name: `lstm_fused_cell/BlockLSTM:1`, shape: [1x2048], format: [BxF],
   where: 
    
    - B - batch size
    - F - number of features in the hidden state `c`
    
    > Note: The previous hidden state will be the initial state for next iteration.  
    

4. Output text, name: `null`, shape: [1xN], format: [BxT],
   where: 
    
    - B - batch size
    - T - number of output text

    > Note: The length of output text is depend on the result of ctc beam search decoder.

### Converted Model

1. Text, name: `Softmax`, shape: [Nx1xM], format: [IxBxL]
   where: 
    
    - I - number of iteration
    - B - batch size
    - L - number of label
    
    > Note: The number of iteration is depend on the feature length. To make the model can execute with different length of input feature, the model will be iterated since the input shape was fixed.
    
2. Hidden state `h`, name: `lstm_fused_cell/BlockLSTM/TensorIterator.1`, shape: [1x2048], format: [BxF],
   where: 
    
    - B - batch size
    - F - number of features in the hidden state `h`
    
    > Note: The previous hidden state will be the initial state for next iteration.  
    
3. Hidden state `c`, name: `lstm_fused_cell/BlockLSTM/TensorIterator.2`, shape: [1x2048], format: [BxF],
   where: 
    
    - B - batch size
    - F - number of features in the hidden state `c`
    
    > Note: The previous hidden state will be the initial state for next iteration.  
    

4. Output text, name: `null`, shape: [1xN], format: [BxT],
   where: 
    
    - B - batch size
    - T - number of output text

    > Note: The length of output text is depend on the result of ctc beam search decoder.

## Legal Information

[https://raw.githubusercontent.com/mozilla/DeepSpeech/v0.5.0/LICENSE]()
