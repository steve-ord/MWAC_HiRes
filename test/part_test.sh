cd ./01
#size=32768

#dd if=../ch_100.dat of=0_100.dat bs=$size count=10000
#dd if=../ch_100.dat of=1_100.dat bs=$size skip=10000 count=10000
#dd if=../ch_100.dat of=2_100.dat bs=$size skip=20000 count=10000
#dd if=../ch_100.dat of=3_100.dat bs=$size skip=30000 count=10000
#dd if=../ch_100.dat of=4_100.dat bs=$size skip=40000 count=10000
#dd if=../ch_100.dat of=5_100.dat bs=$size skip=50000 count=10000
#dd if=../ch_100.dat of=6_100.dat bs=$size skip=60000 count=10000

#cp ../psrfits_header.txt ./
#python ../make_auxfiles.py
dir=`pwd`

rm *.ar
rm *.fits

aprun make_beam -e dat -i -f psrfits_header.txt -d $dir -n 128 -a 128 -r 10000 -o xx -w flags.txt -c phases.txt -t 1 -D $dir/../
dspsr  -c 0.0064 G0024_1118168248_01_0001.fits 
mv 2015-06-12-18\:17\:11.ar incoherent.ar
rm G0024_1118168248_01_0001.fits
aprun make_beam -e dat -f psrfits_header.txt -d $dir -n 128 -a 128 -r 10000 -o xx -w flags.txt -c phases.txt -t 1 -D $dir/../
dspsr -c 0.0064 G0024_1118168248_01_0001.fits
mv 2015-06-12-18\:17\:11.ar coherent.ar
rm G0024_1118168248_01_0001.fits
aprun make_beam -e dat -v psrfits_header.txt -d $dir -n 128 -a 128 -r 10000 -o xx -w flags.txt -c phases.txt -t 1 -D $dir/../
dspsr -c 0.0064 -D 0.0 G0024_1118168248_01.hdr
