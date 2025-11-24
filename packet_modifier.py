from scapy.layers.l2 import Ether
from scapy.utils import PcapReader, wrpcap
import random, string, time, platform, os
from datetime import datetime
from tqdm import tqdm

# Generatore di stringhe random di lunghezza random tra lower e upper
def randStr(chars = string.ascii_uppercase + string.digits, lower=0, upper=100):
    if upper<0:
        upper = 0
    padding = random.randint(lower, upper)
    return ''.join(random.choice(chars) for _ in range(padding))

filters = [] # protocolli da escludere
max_len = 1500
trigger = 100000
min_padding = 1
small_padding = 100
large_padding = 1024
min_size = 0
large_size = 100

if platform.system() == "Windows":
    fs = "\\"
else:
    fs = "/"

def byte_udp(input_file, output_file, lower=0, upper=100, debug = False, max_size = 65000):
    input_pcap_size = int(os.path.getsize(input_file) / (1024*1024))
    print(f"Reading {input_pcap_size} MB from {input_file}")
    read = 0
    manipulated = 0
    with PcapReader(input_file) as pcap:
        print("Manipulating...")
        new_packets = list([])
        for pkt in pcap:
            read += 1
            if ((read%trigger) == 0):
                print(f"Reading {int(read)} packets")
            
            # Gestione dei tipi di pacchetti
            if 'error' in pkt.summary() or not pkt.haslayer(Ether):
                continue
            elif 'TCP' in pkt.summary():
                new_packets.append(pkt)
            # Se il pacchetto è UDP, aggiungo del padding ricostruendo il pacchetto (ricalcolo di lunghezza e checksum)
            elif 'UDP' in pkt.summary():
                if len(filters) > 0:
                    if any(f in pkt.summary() for f in filters):
                        continue
                manipulated += 1
                padding = randStr(lower=lower, upper=min((max_size - pkt.len), upper))

                new_pkt = (pkt['Ether'] / padding)
                
                if 'IPv6' in pkt.summary():
                    del new_pkt['IPv6'].len
                    del new_pkt['IPv6'].chksum
                else:
                    del new_pkt['IP'].len
                    del new_pkt['IP'].chksum
                del new_pkt['UDP'].len
                del new_pkt['UDP'].chksum
                new_pkt = Ether(new_pkt.build())
                new_pkt.time = pkt.time
                new_packets.append(new_pkt)
            else:
                new_packets.append(pkt)
    print(f"Total packets read: {read}")
    
    print(f"Writing {len(new_packets)} packets (of which {manipulated} manipulated)")
    wrpcap(output_file, new_packets)

    output_pcap_size = int(os.path.getsize(output_file) / (1024*1024))
    diff = output_pcap_size - input_pcap_size
    print(f"Wrote {output_pcap_size} MB (diff: {diff})")

def byte_tcp(input_file, output_file, lower=0, upper=100, debug = False, max_size = 65000):
    input_pcap_size = int(os.path.getsize(input_file) / (1024*1024))
    print(f"Reading {input_pcap_size} MB from {input_file}")
    read = 0
    manipulated = 0
    with PcapReader(input_file) as pcap:
        print("Manipulating...")
        new_packets = list([])
        for pkt in pcap:
            read += 1
            if ((read%trigger) == 0):
                print(f"Reading {int(read)} packets")
            
            # Gestione dei tipi di pacchetti
            if 'error' in pkt.summary() or not pkt.haslayer(Ether):
                continue
            elif 'UDP' in pkt.summary():
                new_packets.append(pkt)
            # Se il pacchetto è TCP, aggiungo del padding ricostruendo il pacchetto (ricalcolo di lunghezza e checksum)
            elif 'TCP' in pkt.summary():
                # Se è PUSH, non lo modifico
                if 'P' in pkt['TCP'].flags:
                    new_packets.append(pkt)
                    continue

                if len(filters) > 0:
                    if any(f in pkt.summary() for f in filters):
                        continue
                manipulated += 1
                padding = randStr(lower=lower, upper=min((max_size - pkt.len), upper))

                new_pkt = (pkt['Ether'] / padding)
                
                if 'IPv6' in pkt.summary():
                    del new_pkt['IPv6'].len
                    del new_pkt['IPv6'].chksum
                else:
                    del new_pkt['IP'].len
                    del new_pkt['IP'].chksum
                del new_pkt['UDP'].len
                del new_pkt['UDP'].chksum
                new_pkt = Ether(new_pkt.build())
                new_pkt.time = pkt.time
                new_packets.append(new_pkt)
            else:
                new_packets.append(pkt)
    print(f"Total packets read: {read}")
    
    print(f"Writing {len(new_packets)} packets (of which {manipulated} manipulated)")
    wrpcap(output_file, new_packets)

    output_pcap_size = int(os.path.getsize(output_file) / (1024*1024))
    diff = output_pcap_size - input_pcap_size
    print(f"Wrote {output_pcap_size} MB (diff: {diff})")

input_folder = "notebook/snippet/original/"
output_folder = "notebook/snippet/adversarial/"

for malware in tqdm(os.listdir(input_folder)):
    print(malware)
    for original_pcap in tqdm(os.listdir(input_folder + fs + malware)):
        attack = "-Ts"
        n = original_pcap.replace(".pcap", attack+".pcap")
        input_file = input_folder + malware + fs + original_pcap
        output_file = output_folder + malware + fs + n
        input_pcap_size = int(os.path.getsize(input_file) / (1024*1024))
        if (min_size <= input_pcap_size < large_size): 
            new_packets = byte_udp(input_file, output_file, lower = min_padding, upper = small_padding)
