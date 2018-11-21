cp -rv /ece408_src/* /mxnet/src/operator/custom
cd ece408_project
for src in ece408_src/*; do cp -v $src /mxnet/src/operator/custom/.; done
nice -n20 make -C /mxnet
/usr/bin/time python m3.1.py 100
cd ..
