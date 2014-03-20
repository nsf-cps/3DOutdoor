#!/bin/bash

cd /sys/class/gpio
echo 37 > export
cd gpio37
echo out > direction

for i in {1..20..1}
do
	echo 1 > value
	sleep 1
	echo 0 > value
	echo "$i"
done