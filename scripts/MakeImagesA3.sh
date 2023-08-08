#!/bin/bash

rawdir=${1:-../data}
imgdir=${2:-../data2}
nb_sample=${3:-10}
cores=${4:-1}
imgsize=${5:-100}
type=${6:-png}

#WFPLOT=$HOME/Projects/MachineLearning/WFplot/WFplot.GF
WFPLOT=/storage/disqs/ML-Anderson3D/ML-Anderson3D/WFplot/WFplot.GF
#WFPLOT=$MLdir"/../WFplot/WFplot.GF"
#WFPLOT=/media/phsht/DataDrive/MachineLearning/Anderson/WFplot/WFplot.GF

MAKEDISQS=/storage/disqs/MakeDisQS.sh

nb_sample_raw=5000

codedir=`pwd`

echo "PERCO: rawdir=" $rawdir ", nb_sample=" $nb_sample ", cores=" $cores

mkdir -p $imgdir

# --- working in RAWDIR directories
cd $rawdir
#cd "L"$size"-"$nb_sample_raw
pwd

for Wdir in A3*
do
    echo '>  START with' $rawdir/$Wdir
    
    if [ ! -d $imgdir\$Wdir ]; then
        echo ">-- making "$imgdir/$Wdir
        mkdir -p $imgdir/$Wdir
    else
        echo ">-- cd "$imgdir/$Wdir
        #cd $imgdir"/"$img_folder$Wdir
    fi

    jobfile=MakeImgBASH-`basename $rawdir`".sh"
    imgscript=MakeImgCONVERT-`basename $rawdir`".sh"

    echo ">-- creating "$jobfile + $imgscript "in "$imgdir/$Wdir

cat > $imgdir/$Wdir/${imgscript} << EOD
#!/usr/bin/bash

rawdir=\${1:-../data}
imgdir=\${2:-../data2}
evec=\${3:-"test.raw"}
imgsize=\${4:-100}
color=\${5:-1}
frame=\${6:-0}
type=\${7:-png}

#echo "--> working on making" \$evec "from" \$rawdir "into" \$imgdir "for size" \$imgsize
#echo "--> with color" \$color "frame" \$frame "and type" \$type 

#echo "--> working in" `pwd` "with" \$evec

echo -ne "\$evec\n\$color\n\$frame" | $WFPLOT

epsevec=\`basename \$evec.eps\`
echo \$epsevec

mv \$evec.eps \$imgdir/$epsevec
cd \$imgdir
#pwd

if [ \$imgsize -lt 1 ]
then
    convert \$epsevec \`basename \$epsevec .raw.eps\`.\$type
else
    convert -resize \$imgsize"x"\$imgsize! \$epsevec \`basename \$epsevec .raw.eps\`.\$type
fi

rm -f \$epsevec # delete the .eps file
EOD

cat > $imgdir/$Wdir"/"${jobfile} << EOD
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$cores
#SBATCH --mem-per-cpu=3700

# DT2018
#module load GCCcore/11.2.0 parallel/20210722
#module load Anaconda3
# DT2022
module load GCCcore/11.2.0 parallel/20210722
#module load Python/3.9.6

#conda init --all; conda activate

echo "->- working in Wdir=$rawdir/$Wdir"

cd $rawdir/$Wdir
#echo '->- working in '`pwd`
rm -f raw.lst

for rawfile in \`find . -name \Evec*.raw | sort | head -$nb_sample\`
do
    #echo $imgdir/$Wdir/\`basename \$rawfile .raw\`.$type
    #echo \`dirname \$rawfile\`
    if [ ! -f $imgdir/$Wdir/\`basename \$rawfile .raw\`.$type ]; then
	echo '->-' \$rawfile "needs .$type file"
        #echo "file" \`basename \$rawfile .raw\`.img  "does not exist" 
	echo \$rawfile >> raw.lst
    fi
done

pwd

if [ -f raw.lst ]; then
   sort -R raw.lst | parallel --bar -j$cores -a - source $imgdir/$Wdir/$imgscript $rawdir/$Wdir $imgdir/$Wdir {} $imgsize 1 0 $type
else
   echo '->- all $nb_sample chosen .raw files already converted into .img files --- skipping!'
fi

#if [ ! -f $rawdir/$raw_folder/$Wdir/raw.lst ]; then
#    echo "all the files are already converted to img"
#else
#    echo "the Wdir already exists"
#    sort -R raw.lst | parallel -j$cores -a - source $imgscript {} $Wdir
#fi

pwd

echo "--> finished in Wdir=$Wdir"

EOD

    #cat ${jobfile}
    #cat ${imgscript}

    chmod 755 $imgdir"/"$Wdir"/"${jobfile}
    chmod g+w $imgdir"/"$Wdir"/"${jobfile}

    echo ">-- starting the .sh file"
    #(sbatch -q devel ${jobfile})
    #(sbatch -q taskfarm ${jobfile})
    sbatch ${imgdir}/$Wdir/${jobfile}
    #(${imgdir}/$Wdir/${jobfile})
    
done

cd $codedir

$MAKEDISQS $imgdir
