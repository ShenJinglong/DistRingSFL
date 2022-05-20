#!/bin/bash

# ip link set wlan1 down
# iw dev wlan1 interface add mesh0 type mp
# ifconfig -a | grep mesh0
# iw dev mesh0 set channel 44
# ip link set mesh0 down
# iw dev mesh0 set meshid ringsfl
# ifconfig mesh0 up
# ifconfig mesh0 10.0.0.1 netmask 255.255.255.0

systemctl stop NetworkManager
iw reg set CN
ip link set wlx0013ef4f0fdf down
iw dev wlx0013ef4f0fdf interface add ibss0 type ibss
ip link set up mtu 1532 dev ibss0
iw dev ibss0 ibss join ringsfl 5220 HT40+ fixed-freq 02:12:34:56:78:9A
batctl if add ibss0
batctl if
ifconfig bat0 10.0.0.1 netmask 255.255.255.0
