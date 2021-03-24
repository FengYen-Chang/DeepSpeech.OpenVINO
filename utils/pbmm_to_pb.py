#! /usr/bin/env python3
#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse

from pathlib import Path

from tensorflow.compat.v1 import GraphDef

from mds_convert_utils.memmapped_file_system_pb2 import MemmappedFileSystemDirectory

import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert memory-mapped Tensorflow protobuf format into plain Tensorflow protobuf (GraphDef)")
    parser.add_argument('input', type=Path, help="Input .pbmm filename")
    parser.add_argument('output', type=Path, help="Output .pb filename")
    return parser.parse_args()


def load_memmapped_fs(data):
    data_directory_ofs = int.from_bytes(data[-8:], 'little')
    data_directory = data[data_directory_ofs:-8]

    directory_pb = MemmappedFileSystemDirectory()
    directory_pb.ParseFromString(data_directory)

    return {
        entry.name: data[entry.offset : entry.offset + entry.length]
        for entry in directory_pb.element
    }


def convert_node(node, mmfs):
    """
    node is changed in place.
    """
    node.op = 'Const'
    # region_name is bytes here, while it is string in MemmappedFileSystemDirectory
    region_name = node.attr['memory_region_name'].s.decode('utf-8')
    dtype = node.attr['dtype'].type
    shape = node.attr['shape'].shape
    del node.attr['memory_region_name']
    del node.attr['shape']
    # keep node.attr['dtype']
    node.attr['value'].tensor.dtype = dtype
    node.attr['value'].tensor.tensor_shape.CopyFrom(shape)
    node.attr['value'].tensor.tensor_content = mmfs[region_name]


def undo_mmap(graph_def, mmfs):
    """
    graph_def is changed in place.
    """
    for node in graph_def.node:
        if node.op == 'ImmutableConst':
            convert_node(node, mmfs)

def replace_string_val(node, temp_node):
    node.attr['value'].tensor.string_val[:] = temp_node.attr['value'].tensor.string_val[:]

def replace_metadata_alphabet(graph_def):
    """
    remove alphabet metadata to avoid upexpected ascii code. 
    """
    temp_node = None
    for node in graph_def.node:
        if node.name == 'metadata_language':
            temp_node = node
    
    for node in graph_def.node:
        if node.name == 'metadata_alphabet':
            replace_string_val(node, temp_node)

def main():
    args = parse_args()

    data = args.input.read_bytes()

    mmfs = load_memmapped_fs(data)
    ROOT = 'memmapped_package://.'

    graph_def = GraphDef()
    graph_def.ParseFromString(mmfs[ROOT])

    undo_mmap(graph_def, mmfs)
    
    replace_metadata_alphabet(graph_def)

    args.output.write_bytes(graph_def.SerializeToString())

    
    

if __name__ == '__main__':
    main()
