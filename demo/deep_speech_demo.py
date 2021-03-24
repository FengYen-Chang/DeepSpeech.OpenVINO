#!/usr/bin/env python
"""
 Copyright (C) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore

import wave
import codecs

from speech_features import audio_spectrogram, mfcc 
from ctc_beamsearch_decoder import ctc_beam_search_decoder


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to an audio file.",
                      required=True,
                      type=str)
    args.add_argument("-l", "--language", help="Required. Define the trained model is for CN or ENG.",
                      required=True,
                      type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)

    return parser


n_input    = 26
n_context  = 9
n_steps    = 16
numcep     = n_input
numcontext = n_context
beamwidth  = 10

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    
    if args.language == 'CN':
        alphabet = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '0A', '0B', '0C', '0D', '0E', '0F', 
                    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '1A', '1B', '1C', '1D', '1E', '1F', 
                    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '2A', '2B', '2C', '2D', '2E', '2F', 
                    '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '3A', '3B', '3C', '3D', '3E', '3F', 
                    '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '4A', '4B', '4C', '4D', '4E', '4F', 
                    '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '5A', '5B', '5C', '5D', '5E', '5F', 
                    '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '6A', '6B', '6C', '6D', '6E', '6F', 
                    '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '7A', '7B', '7C', '7D', '7E', '7F', 
                    '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '8A', '8B', '8C', '8D', '8E', '8F', 
                    '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '9A', '9B', '9C', '9D', '9E', '9F', 
                    'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 
                    'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 
                    'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 
                    'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'DA', 'DB', 'DC', 'DD', 'DE', 'DF', 
                    'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'EA', 'EB', 'EC', 'ED', 'EE', 'EF', 
                    'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'FA', 'FB', 'FC', 'FD', 'FE', 'FF',
                    '_'
        ]
    elif args.language == 'ENG':
        alphabet = " abcdefghijklmnopqrstuvwxyz'-"
    else :
        log.error("Please use -l or --language to assign language.")

    # Speech feature extration
    _wave = wave.open(args.input, 'rb')
    fs = _wave.getframerate()
    
    if fs != 16000:
        log.error("Please using 16kHz wave file, not {}Hz\n".format(fs))
    _length = _wave.getnframes()
    audio = np.frombuffer(_wave.readframes(_length), dtype=np.dtype('<h'))

    audio = audio/np.float32(32768) # normalize to -1 to 1, int 16 to float32
 
    audio = audio.reshape(-1, 1)
    spectrogram = audio_spectrogram(audio, (16000 * 32 / 1000), (16000 * 20 / 1000), True)
    features = mfcc(spectrogram.reshape(1, spectrogram.shape[0], -1), fs, 26)

    empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
    features = np.concatenate((empty_context, features, empty_context))    

    num_strides = len(features) - (n_context * 2)
    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2*n_context+1
    features = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, n_input),
        (features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    assert len(net.inputs.keys()) == 3, "Sample supports only three input topologies"
    assert len(net.outputs) == 3, "Sample supports only three output topologies"

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    state_h = np.zeros((1, 2048))
    state_c = np.zeros((1, 2048))
    logits = np.empty([0, 1, len(alphabet)])

    for i in range(0, len(features), n_steps):
        chunk = features[i:i+n_steps]
        
        if len(chunk) < n_steps:
            chunk = np.pad(chunk,
                           (
                            (0, n_steps - len(chunk)),
                            (0, 0),
                            (0, 0)
                           ),
                           mode='constant',
                           constant_values=0)

        res = exec_net.infer(inputs={'previous_state_c': state_c,
                                     'previous_state_h': state_h,
                                      'input_node': [chunk]})
                                      
        # Processing output blob
        logits = np.concatenate((logits, res['logits']))
        state_h = res['cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/BlockLSTM/TensorIterator.1']
        state_c = res['cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/BlockLSTM/TensorIterator.2']

    if args.language == 'CN':
        output_byte = bytes.fromhex(ctc_beam_search_decoder(logits, alphabet, alphabet[-1], beamwidth))
        print ("\n>>>{}\n".format(output_byte.decode('utf-8')))
    if args.language == 'ENG':
        print ("\n>>>{}\n".format(ctc_beam_search_decoder(logits, alphabet, alphabet[-1], beamwidth)))

if __name__ == '__main__':
    sys.exit(main() or 0)

