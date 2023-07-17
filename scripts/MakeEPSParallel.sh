#!/bin/bash

size=${1:-20}
nsamples=${2:-2}
color=${3:-1}
frame=${4:-0}
type=${5:-jpg}
imgsize=${6:-0}

echo $size $nsamples $type

#datadir="/media/phsht/DataDrive/AML3D_data"
datadir="/storage/disqs/phsht/Archive-DATA/MULTIFRACTALS/AML3D_data"
#datadir="/mnt/md0/phsht/data/AML3D_data"

MLdir=`pwd`
echo $MLdir

#WFPLOT=$HOME/Projects/MachineLearning/WFplot/WFplot.GF
WFPLOT=/storage/disqs/MachineLearning-Anderson/WFplot/WFplot.sh
#WFPLOT=$MLdir"/../WFplot/WFplot.GF"
#WFPLOT=/media/phsht/DataDrive/MachineLearning/Anderson/WFplot/WFplot.GF

# copy the original data files

cd $datadir
pwd

for disdir in W*/
do

    echo $disdir
    #cd $disdir

    for dir in `ls -d $disdir/L$size/AM-* | head -$nsamples`
    do
	echo $dir
	mkdir -p $MLdir/$disdir
	cp -u --preserve=timestamps $dir/Evec*.bz2 $MLdir/$disdir
    done

    #cd ..
done

# make images from the copied data files

cd $MLdir
pwd

for disdir in W*/
do

    echo $disdir
    cd $disdir
    #bunzip2 -f *$size*.bz2
    find . -name "*.raw.bz2" | parallel -I% --max-args 1 bunzip2 %

#    for evec in Evec*.raw
#    do
#	echo $evec
#	echo -ne "$evec\n$color\n$frame" | $WFPLOT
#    done

#	if [ $imgsize -lt 1 ]
#	then
#	    convert $evec.eps `basename $evec .raw.eps`.$type
#	else
#	    convert -resize $imgsize"x"$imgsize\! $evec.eps `basename $evec .raw.eps`.$type
#	fi

#    find . -name "*.raw" | parallel -I% --max-args 1 (echo -ne "%\n$color\n$frame" | $WFPLOT)

    find . -name "*.raw" | parallel -I% --max-args 1 $WFPLOT % $color $frame

#    if [ $imgsize -lt 1 ]
#    then
#	find . -name "*.raw.eps" | parallel -I% --max-args 1 convert % `basename % .eps`.$type
#    else
#    	find . -name "*.raw.eps" | parallel -I% --max-args 1 convert -resize $imgsize"x"$imgsize\! % `basename % .eps`.$type
#    fi
#    rename .raw.eps.jpg .jpg *.jpg

#    rm -f $evec.eps $evec
#    find . -name "*.raw.eps" | parallel -I% --max-args 1 rm -f %
    find . -name "*.raw" | parallel -I% --max-args 1 rm -f %

#    done
    cd ..
done
    

