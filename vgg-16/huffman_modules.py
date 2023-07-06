import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
from pathlib import Path

import torch
from scipy.sparse import csr_matrix, csc_matrix

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq

conv_layers_ori = ['features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21', 'features.24', 'features.26', 'features.28']
conv_layers_exit = ['head.0', 'head.2', 'head.5', 'head.7', 'head.10', 'head.12', 'head.14', 'backbone.17', 'backbone.19', 'backbone.21', 'backbone.24', 'backbone.26', 'backbone.28']

class huffman_Linear(Module):
    def __init__(self, module):
        super(huffman_Linear, self).__init__()
        dev = module.weight.device
        codebook = module.codebook.data
        self.codebook = Parameter(codebook, requires_grad=False)
        self.bias = Parameter(module.bias.data)
        weight = module.weight.data.cpu().numpy()
        print(weight)
        self.shape = weight.shape
        weight_tensor, hftree_tenosr = self.huffman_encode(weight.reshape(-1))
        # print(weight_tensor)
        # print(hftree_tenosr)
        self.weight = Parameter(weight_tensor.type(torch.uint8).to(dev), requires_grad=False)
        self.hftree = Parameter(hftree_tenosr.type(torch.uint8).to(dev), requires_grad=False)


    def huffman_encode(self, arr):
        # dtype = str(arr.dtype)

        # Calculate frequency in arr
        # print(arr)
        freq_map = defaultdict(int)

        for value in np.nditer(arr):
            value = int(value)
            freq_map[value] += 1

        # Make heap
        heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
        heapify(heap)

        # Merge nodes
        while len(heap) > 1:
            node1 = heappop(heap)
            node2 = heappop(heap)
            merged = Node(node1.freq + node2.freq, None, node1, node2)
            heappush(heap, merged)

        # Generate code value mapping
        value2code = {}

        def generate_code(node, code):
            if node is None:
                return
            if node.value is not None:
                value2code[node.value] = code
                return
            generate_code(node.left, code + '0')
            generate_code(node.right, code + '1')

        root = heappop(heap)
        generate_code(root, '')

        # Dump data
        code_str = ''.join(value2code[int(value)] for value in np.nditer(arr))
        num_of_padding = -len(code_str) % 8
        header = f"{num_of_padding:08b}"
        code_str = header + code_str + '0' * num_of_padding

        # Convert string to integers and to real bytes
        byte_arr = np.array([int(code_str[i:i + 8], 2) for i in range(0, len(code_str), 8)])

        code_list = []

        def int2bitstr(integer):
            four_bytes = struct.pack('>I', integer)  # bytes
            return ''.join(f'{byte:08b}' for byte in four_bytes)  # string of '0's and '1's

        def encode_node(node):
            if node.value is not None:  # node is leaf node
                code_list.append('1')
                lst = list(int2bitstr(node.value))
                code_list.extend(lst)
            else:
                code_list.append('0')
                encode_node(node.left)
                encode_node(node.right)

        encode_node(root)
        codebook_str = ''.join(code_list)
        num_of_padding = -len(codebook_str) % 8
        header = f"{num_of_padding:08b}"
        codebook_str = header + codebook_str + '0' * num_of_padding
        # Convert string to integers and to real bytes
        code_arr = np.array([int(codebook_str[i:i + 8], 2) for i in range(0, len(codebook_str), 8)])
        print(byte_arr.size)
        print(code_arr.size)
        return torch.from_numpy(byte_arr), torch.from_numpy(code_arr)


    def huffman_decode(self):
        # Read the codebook
        codebook_arr = self.hftree.data.cpu().numpy().tolist()
        header = codebook_arr[0]
        rest = codebook_arr[1:]  # bytes
        codebook_str = ''.join(f'{byte:08b}' for byte in rest)
        offset = header
        if offset != 0:
            codebook_str = codebook_str[:-offset]

        def bitstr2int(bitstr):
            byte_arr = bytearray(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
            return struct.unpack('>I', byte_arr)[0]

        def decode_huffman_tree(code_str):
            """
            Decodes a string of '0's and '1's and costructs a huffman tree
            """
            idx = 0

            def decode_node():
                nonlocal idx
                info = code_str[idx]
                idx += 1
                if info == '1':  # Leaf node
                    value = bitstr2int(code_str[idx:idx + 32])
                    idx += 32
                    return Node(0, value, None, None)
                else:
                    left = decode_node()
                    right = decode_node()
                    return Node(0, None, left, right)

            return decode_node()

        root = decode_huffman_tree(codebook_str)


        # Read the data
        data_arr = self.weight.data.cpu().numpy().tolist()
        header = data_arr[0]
        rest = data_arr[1:]  # bytes
        data_str = ''.join(f'{byte:08b}' for byte in rest)
        offset = header
        if offset != 0:
            data_str = data_str[:-offset]
        # Decode
        data = []
        ptr = root
        for bit in data_str:
            ptr = ptr.left if bit == '0' else ptr.right
            if ptr.value is not None:  # Leaf node
                data.append(ptr.value)
                ptr = root

        print(np.array(data, dtype=np.uint8).reshape(self.shape))
        self.weight = Parameter(torch.from_numpy(np.array(data, dtype=np.uint8).reshape(self.shape)).to(self.weight.device), requires_grad=False)


    def dequantize(self):
        weight = self.weight.data.cpu().numpy()
        codebook = self.codebook.data.cpu().numpy()
        dev = self.weight.device
        self.weight.data = torch.from_numpy(codebook[weight.reshape(-1)].reshape(self.shape)).type(torch.float32).to(dev)


    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


# Huffman Coding for Conv Layers
def huffman_Conv(model):
    for name, module in model.named_modules():
        if name in conv_layers_exit:
            dev = module.weight.device
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape
            # print(weight)
            weight_tensor, hftree_tenosr = huffman_encode(weight.reshape(-1))
            module.weight = Parameter(weight_tensor.type(torch.uint8).to(dev), requires_grad=False)
            module.register_parameter('hftree', Parameter(hftree_tenosr.type(torch.uint8).to(dev), requires_grad=False))
            module.register_parameter('shape', Parameter(torch.from_numpy(np.array(shape)).to(dev), requires_grad=False))




def huffman_decoding(model):
    for name, module in model.named_modules():
        if name in conv_layers_exit:
            huffman_decode(module)



def huffman_encode(arr):
    freq_map = defaultdict(int)

    for value in np.nditer(arr):
        value = int(value)
        freq_map[value] += 1

    # Make heap
    heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
    heapify(heap)

    # Merge nodes
    while len(heap) > 1:
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = Node(node1.freq + node2.freq, None, node1, node2)
        heappush(heap, merged)

    # Generate code value mapping
    value2code = {}

    def generate_code(node, code):
        if node is None:
            return
        if node.value is not None:
            value2code[node.value] = code
            return
        generate_code(node.left, code + '0')
        generate_code(node.right, code + '1')

    root = heappop(heap)
    generate_code(root, '')

    # Dump data
    code_str = ''.join(value2code[int(value)] for value in np.nditer(arr))
    num_of_padding = -len(code_str) % 8
    header = f"{num_of_padding:08b}"
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = np.array([int(code_str[i:i + 8], 2) for i in range(0, len(code_str), 8)])

    code_list = []

    def int2bitstr(integer):
        four_bytes = struct.pack('>I', integer)  # bytes
        return ''.join(f'{byte:08b}' for byte in four_bytes)  # string of '0's and '1's

    def encode_node(node):
        if node.value is not None:  # node is leaf node
            code_list.append('1')
            lst = list(int2bitstr(node.value))
            code_list.extend(lst)
        else:
            code_list.append('0')
            encode_node(node.left)
            encode_node(node.right)

    encode_node(root)
    codebook_str = ''.join(code_list)
    num_of_padding = -len(codebook_str) % 8
    header = f"{num_of_padding:08b}"
    codebook_str = header + codebook_str + '0' * num_of_padding
    # Convert string to integers and to real bytes
    code_arr = np.array([int(codebook_str[i:i + 8], 2) for i in range(0, len(codebook_str), 8)])
    print(byte_arr.size)
    print(code_arr.size)
    return torch.from_numpy(byte_arr), torch.from_numpy(code_arr)



def huffman_decode(module):
    # Read the codebook
    codebook_arr = module.hftree.data.cpu().numpy().tolist()
    header = codebook_arr[0]
    rest = codebook_arr[1:]  # bytes
    codebook_str = ''.join(f'{byte:08b}' for byte in rest)
    offset = header
    if offset != 0:
        codebook_str = codebook_str[:-offset]

    def bitstr2int(bitstr):
        byte_arr = bytearray(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
        return struct.unpack('>I', byte_arr)[0]

    def decode_huffman_tree(code_str):
        """
        Decodes a string of '0's and '1's and costructs a huffman tree
        """
        idx = 0

        def decode_node():
            nonlocal idx
            info = code_str[idx]
            idx += 1
            if info == '1':  # Leaf node
                value = bitstr2int(code_str[idx:idx + 32])
                idx += 32
                return Node(0, value, None, None)
            else:
                left = decode_node()
                right = decode_node()
                return Node(0, None, left, right)

        return decode_node()

    root = decode_huffman_tree(codebook_str)


    # Read the data
    data_arr = module.weight.data.cpu().numpy().tolist()
    header = data_arr[0]
    rest = data_arr[1:]  # bytes
    data_str = ''.join(f'{byte:08b}' for byte in rest)
    offset = header
    if offset != 0:
        data_str = data_str[:-offset]
    # Decode
    data = []
    ptr = root
    for bit in data_str:
        ptr = ptr.left if bit == '0' else ptr.right
        if ptr.value is not None:  # Leaf node
            data.append(ptr.value)
            ptr = root

    # print(np.array(data, dtype=np.uint8).reshape(module.shape))
    shape = tuple(module.shape.data.cpu().numpy())

    module.weight = Parameter(torch.from_numpy(np.array(data, dtype=np.uint8).reshape(shape)).to(module.weight.device), requires_grad=False)




