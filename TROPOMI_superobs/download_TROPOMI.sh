#!/bin/sh

# ###########################
# Download TROPOMI L2 dataset
# ###########################

for mth in {12..12}
do
    for day in {1..31}
    do
	filetest="s3://DIAS/Sentinel-5P/TROPOMI/L2__CO____/2018/$(printf "%02d" $mth)/$(printf "%02d" $day)/"
	echo $filetest
	s3cmd --recursive -c ~/.s3cfg ls $filetest > test.txt
	sed '/NRTI/d' ./test.txt > test0.txt
	sed '/.cdl/d' ./test0.txt > test01.txt
	sed 's#^.\{31\}#s3cmd --recursive -c ~/.s3cfg get #g' test01.txt > test2.txt
	sed 's#$# /nobackupp19/bbyrne1/Test_TROPOMI_download/PRODUCT/#g' test2.txt > test3.sh
	sed -i '1s*^*#!/bin/sh\n*' test3.sh
	chmod u+x test3.sh
	./test3.sh
    done
done
