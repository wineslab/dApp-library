#!/bin/bash

sudo tcpdump -i lo -nn -vv -s 0 -w sctp_traffic.pcap 'port 5000 and ip[9] == 132'