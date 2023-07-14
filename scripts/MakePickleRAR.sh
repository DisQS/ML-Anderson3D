#!/bin/bash

rawdir=${1:-../data}
pkldir=${2:-../data2}
nb_sample=${3:-10}
cores=${4:-1}

MAKEDISQS=/storage/disqs/MakeDisQS.sh

nb_sample_raw=5000

codedir=`pwd`

echo "PERCO: rawdir=" $rawdir ", nb_sample=" $nb_sample ", cores=" $cores

mkdir -p $pkldir

# --- working in RAWDIR directories
cd $rawdir
#cd "L"$size"-"$nb_sample_raw
pwd

for Wdir in A3*
do
    echo '>  START with' $rawdir/$Wdir
    
    if [ ! -d $pkldir\$Wdir ]; then
        echo ">-- making "$pkldir/$Wdir
        mkdir -p $pkldir/$Wdir
    else
        echo ">-- cd "$pkldir/$Wdir
        #cd $pkldir"/"$pkl_folder$Wdir
    fi

    jobfile=MakePickle-`basename $rawdir`".sh"
    pyscript=MakePickle-`basename $rawdir`".py"

    echo ">-- creating "$jobfile + $pyscript "in "$pkldir/$Wdir

cat > $pkldir/$Wdir/${pyscript} << EOD
#!/usr/bin/python3
import sys
import numpy as np
import os
import pickle as pkl

#print(sys.argv)

filepath = os.path.dirname(str(sys.argv[3]))
basename = os.path.basename(str(sys.argv[3]))
#print('fp:', filepath)
#print('bn:', basename)

filename, file_extension = os.path.splitext(basename)
print('--> creation of file',filename+'.pkl')

raw_path=str(sys.argv[1])+filepath[1:]+'/'+filename+'.raw'
pkl_path=str(sys.argv[2])+'/'+filename+'.pkl'

#print('raw: ',raw_path)
#print('pkl: ',pkl_path)

#print(sys.argv[1])
data=np.loadtxt(str(raw_path))

pickle_file = open(pkl_path,"wb")
pkl.dump(data, pickle_file)
pickle_file.close()
EOD

cat > $pkldir/$Wdir"/"${jobfile} << EOD
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3700

# DT2018
#module load GCCcore/11.2.0 parallel/20210722
#module load Anaconda3
# DT2022
module load GCCcore/11.2.0 parallel/20210722
module load Python/3.9.6

#conda init --all; conda activate

echo "->- working in Wdir=$rawdir/$Wdir"

cd $rawdir/$Wdir
#echo '->- working in '`pwd`
rm -f raw.lst

for rawfile in \`find . -name \Evec*.raw | sort | head -$nb_sample\`
do
    #echo $pkldir/$Wdir/\`basename \$rawfile .raw\`.pkl
    #echo \`dirname \$rawfile\`
    if [ ! -f $pkldir/$Wdir/\`basename \$rawfile .raw\`.pkl ]; then
	echo '->-' \$rawfile "needs .pkl file"
        #echo "file" \`basename \$rawfile .raw\`.pkl  "does not exist" 
	echo \$rawfile >> raw.lst
    fi
done

pwd

if [ -f raw.lst ]; then
   sort -R raw.lst | parallel -j$cores -a - python $pkldir/$Wdir/$pyscript $rawdir/$Wdir $pkldir/$Wdir {}
else
   echo '->- all $nb_sample .raw files already converted into .pkl files --- skipping!'
fi

#if [ ! -f $rawdir/$raw_folder/$Wdir/raw.lst ]; then
#    echo "all the files are already converted to pkl"
#else
#    echo "the Wdir already exists"
#    sort -R raw.lst | parallel -j$cores -a - python $pyscript {} $Wdir
#fi

pwd

echo "--> finished in Wdir=$Wdir"

EOD

    #cat ${jobfile}
    #cat ${pyscript}

    chmod 755 $pkldir"/"$Wdir"/"${jobfile}
    chmod g+w $pkldir"/"$Wdir"/"${jobfile}

    echo ">-- starting the .sh file"
    #(sbatch -q devel ${jobfile})
    #(sbatch -q taskfarm ${jobfile})
    #(sbatch ${jobfile})
    (${pkldir}"/"$pkl_folder$Wdir"/"${jobfile})
    
done

cd $codedir

$MAKEDISQS $pkldir"/"$pkl_folder
