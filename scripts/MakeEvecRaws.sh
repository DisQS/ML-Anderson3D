#!/bin/bash

size=${1:-20}
nsamples=${2:-2}
zip=${3:-0}

echo $size $nsamples $type

#datadir="/media/phsht/DataDrive/AML3D_data"
datadir="/storage/disqs/phsht/Archive-DATA/MULTIFRACTALS/AML3D_data"
#datadir="/mnt/md0/phsht/data/AML3D_data"

MLdir=`pwd`
echo $MLdir

#WFPLOT=$HOME/Projects/MachineLearning/WFplot/WFplot.GF
#WFPLOT=/storage/disqs/Anderson-DB/MachineLearning-Anderson/WFplot/WFplot.GF
#WFPLOT=$MLdir"/../WFplot/WFplot.GF"
#WFPLOT=/media/phsht/DataDrive/MachineLearning/Anderson/WFplot/WFplot.GF

MAKEDISQS=/storage/disqs/MakeDisQS.sh

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
    bunzip2 -f *$size*.bz2

#    for evec in Evec*.raw
#    do
#	echo $evec
#	echo -ne "$evec\n$color\n$frame" | $WFPLOT
#	if [ $imgsize -lt 1 ]
#	then
#	    convert $evec.eps `basename $evec .raw.eps`.$type
#	else
#	    convert -resize $imgsize"x"$imgsize\! $evec.eps `basename $evec .raw.eps`.$type
#	fi
#	rm -f $evec.eps $evec
#   done
    cd ..

    if [ $zip -eq 2 ]; then
	zip -rum `basename $disdir`.zip $disdir
    elif [ $zip -eq 1 ]; then
	zip -ru `basename $disdir`.zip $disdir
    fi
done

cd $MLdir
cd ..
$MAKEDISQS $MLdir

