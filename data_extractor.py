import wget
import requests

url = "https://mcfp.felk.cvut.cz/publicDatasets/"
path = "/home/luca/Desktop/tirocinio/concept_drift_test/lts/malicious/wannacry/"
s = requests.session()

normal_flows = [
    'CTU-Normal-7',
    'CTU-Normal-8-1',
    'CTU-Normal-8-2',
    'CTU-Normal-9',
    # 'CTU-Normal-12',
    'CTU-Normal-13',
    'CTU-Normal-14',
    'CTU-Normal-18',
    'CTU-Normal-20',
    'CTU-Normal-21',
    'CTU-Normal-22',
    'CTU-Normal-23',
    'CTU-Normal-24',
    'CTU-Normal-25',
    'CTU-Normal-26',
    'CTU-Normal-27',
    'CTU-Normal-28',
    'CTU-Normal-29',
    'CTU-Normal-30',
    'CTU-Normal-31',
    # 'CTU-Normal-32',
    # 'CTU-Normal-33'
]
trickbot_flows = [
    "CTU-Malware-Capture-Botnet-238-1",
    "CTU-Malware-Capture-Botnet-239-1",
    "CTU-Malware-Capture-Botnet-240-1",
    "CTU-Malware-Capture-Botnet-241-1",
    "CTU-Malware-Capture-Botnet-242-1",
    "CTU-Malware-Capture-Botnet-243-1",
    "CTU-Malware-Capture-Botnet-244-1",
    "CTU-Malware-Capture-Botnet-247-1",
    "CTU-Malware-Capture-Botnet-261-1",
    "CTU-Malware-Capture-Botnet-261-2",
    "CTU-Malware-Capture-Botnet-265-1",
    "CTU-Malware-Capture-Botnet-266-1",
    "CTU-Malware-Capture-Botnet-267-1",
    "CTU-Malware-Capture-Botnet-273-1",
    "CTU-Malware-Capture-Botnet-324-1",
    "CTU-Malware-Capture-Botnet-325-1",
    "CTU-Malware-Capture-Botnet-327-1",
    "CTU-Malware-Capture-Botnet-327-2"
]
wannacry_flows = [
    "CTU-Malware-Capture-Botnet-252-1",
    "CTU-Malware-Capture-Botnet-253-1",
    "CTU-Malware-Capture-Botnet-254-1",
    "CTU-Malware-Capture-Botnet-256-1",
    "CTU-Malware-Capture-Botnet-258-1",
    "CTU-Malware-Capture-Botnet-270-1",
    "CTU-Malware-Capture-Botnet-283-1",
    "CTU-Malware-Capture-Botnet-284-1",
    "CTU-Malware-Capture-Botnet-285-1",
    "CTU-Malware-Capture-Botnet-286-1",
    "CTU-Malware-Capture-Botnet-287-1",
    "CTU-Malware-Capture-Botnet-294-1",
    "CTU-Malware-Capture-Botnet-295-1",
    "CTU-Malware-Capture-Botnet-296-1",
    "CTU-Malware-Capture-Botnet-297-1"
]
dridex_flows = [
    "CTU-Malware-Capture-Botnet-113-1",
    "CTU-Malware-Capture-Botnet-153-1",
    # "CTU-Malware-Capture-Botnet-218-1",
    "CTU-Malware-Capture-Botnet-227-1",
    "CTU-Malware-Capture-Botnet-228-1",
    "CTU-Malware-Capture-Botnet-246-1",
    "CTU-Malware-Capture-Botnet-248-1",
    "CTU-Malware-Capture-Botnet-249-1",
    "CTU-Malware-Capture-Botnet-257-1",
    "CTU-Malware-Capture-Botnet-259-1",
    "CTU-Malware-Capture-Botnet-260-1",
    "CTU-Malware-Capture-Botnet-263-1",
    "CTU-Malware-Capture-Botnet-322-1",
    "CTU-Malware-Capture-Botnet-326-1",
    "CTU-Malware-Capture-Botnet-346-1"
]
artemis_flows = [
    "CTU-Malware-Capture-Botnet-275-1",
    "CTU-Malware-Capture-Botnet-305-1",
    "CTU-Malware-Capture-Botnet-306-1",
    "CTU-Malware-Capture-Botnet-311-1",
    "CTU-Malware-Capture-Botnet-316-1"
]

for i, n in enumerate(wannacry_flows, 1):
    # Download del binetflow
    tmp = s.get(url + n).text
    pcap = tmp.split(".pcap")[1]
    pcap = pcap[2:] + ".pcap"

    print(pcap)
    wget.download(url + n + "/" + pcap, path + pcap.split("_")[0] + "wannacry" + str(i) + ".pcap")
    