#!/bin/bash

size=${1:-20}
nsamples=${2:-2}
zip=${3:-0}
imgsize=${4:-0}
type=${5:-png}
color=${6:-1}
frame=${7:-0}

# the SCRTP desktop 170523
module load GCCcore/11.2.0 parallel/20210722

echo $size $nsamples $type

#datadir="/media/phsht/DataDrive/AML3D_data"
datadir="/storage/disqs/phsht/Archive-DATA/MULTIFRACTALS/AML3D_data"
#datadir="/mnt/md0/phsht/data/AML3D_data"

MLdir=`pwd`
echo $MLdir

#WFPLOT=$HOME/Projects/MachineLearning/WFplot/WFplot.GF
WFPLOT=/storage/disqs/ML-Anderson3D/ML-Anderson3D/WFplot/WFplot.GF
#WFPLOT=$MLdir"/../WFplot/WFplot.GF"
#WFPLOT=/media/phsht/DataDrive/MachineLearning/Anderson/WFplot/WFplot.GF

MAKEDISQS=/storage/disqs/MakeDisQS.sh

# copy the original data files

cd $datadir
pwd

for disdir in W*/
do
    mkdir -p $MLdir/$disdir # this is where the results will go

    content=`basename $disdir`".txt"
    if [ -e $MLdir/$content ] 
    then
	echo "--- using existing" $content 
    else
	echo "--- creating a FRESH" $content
	ls $datadir/$disdir/L$size/AM-*/*.raw.bz2 > $MLdir/$content
    fi
done

cd $MLdir
pwd

for disdir in W*/
do
    echo "--- bunzip2-ing" $nsamples "samples in" $disdir
    cd $disdir

    head -n$nsamples ../`basename $disdir`'.txt' | parallel -eta 'if [ ! -e {/.} ]; then bzcat --keep {} > {/.}; fi'

    cd ..
done

# make images from the copied data files

cd $MLdir
pwd

for disdir in W*/
do

    echo $disdir
    cd $disdir
    #bunzip2 -f *$size*.bz2

    echo "--- creating EPS images"
    pwd;ls -1 *.raw
    ls -1 *.raw | parallel 'echo -ne "{}\n1\n0\n" | $WFPLOT' 

    echo "--- converting EPS images into" $type
    ls *.eps
    if [ $imgsize -lt 1 ]
    then
	parallel -eta 'convert -density 50 -antialias -colors 128 -background white -normalize -units PixelsPerInch -quality 100 {} `basename {} .raw.eps`.$type' ::: *.eps
    else
	parallel -eta 'convert -density 50 -antialias -colors 128 -background white -normalize -units PixelsPerInch -quality 100 -resize $imgsize"x"$imgsize\! {} `basename {} .raw.eps`.$type' ::: *.eps
	#convert -resize $imgsize"x"$imgsize\! $evec.eps `basename $evec .raw.eps`.$type
    fi

    #rm -f *.eps $evec

    cd ..

    if [ $zip -eq 2 ]; then
	zip -urm `basename $disdir`.zip $disdir
    elif [ $zip -eq 1 ]; then
	zip -ur `basename $disdir`.zip $disdir
    fi
done
    
cd $MLdir
cd ..
$MAKEDISQS $MLdir

