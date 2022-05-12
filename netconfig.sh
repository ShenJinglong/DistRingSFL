#!/bin/bash
ip link set wlan1 down
iw dev wlan1 interface add mesh0 type mp
ifconfig -a | grep mesh0
iw dev mesh0 set channel 13
ip link set mesh0 down
iw dev mesh0 set meshid ringsfl
ifconfig mesh0 up
ifconfig mesh0 10.0.0.2 netmask 255.255.255.0