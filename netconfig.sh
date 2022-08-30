#!/bin/bash

# ip link set wlan1 down
# iw dev wlan1 interface add mesh0 type mp
# ifconfig -a | grep mesh0
# iw dev mesh0 set channel 44
# ip link set mesh0 down
# iw dev mesh0 set meshid ringsfl
# ifconfig mesh0 up
# ifconfig mesh0 10.0.0.1 netmask 255.255.255.0

if [ ${HOSTNAME} == "sjinglong-desktop" ]
then
    systemctl stop NetworkManager
    iw reg set CN
    interfaceName="wlx0013ef4f0fdf"
    ipAddress="10.0.0.1"
elif [ ${HOSTNAME} == "rasp1" ]
then
    interfaceName="wlan1"
    ipAddress="10.0.0.2"
elif [ ${HOSTNAME} == "rasp2" ]
then
    interfaceName="wlan1"
    ipAddress="10.0.0.3"
elif [ ${HOSTNAME} == "rasp3" ]
then
    interfaceName="wlan1"
    ipAddress="10.0.0.4"
fi

ip link set ${interfaceName} down
iw dev ${interfaceName} interface add ibss0 type ibss
ip link set up mtu 1532 dev ibss0
iw dev ibss0 ibss join ringsfl 5220 HT40+ fixed-freq 02:12:34:56:78:9A
batctl if add ibss0
batctl if
ifconfig bat0 ${ipAddress} netmask 255.255.255.0
