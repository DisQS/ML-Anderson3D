#!/bin/bash

dir=${1:-../data}
destdir=${2:-../data2}
size=${3:-10}
nb_sample=${4:-5000}
cores=${5:-1}

MAKEDISQS=/storage/disqs/MakeDisQS.sh

nb_sample_raw=5000

codedir=`pwd`
pkl_folder="L"$size"-"$nb_sample"-pkl/"
raw_folder="L"$size"-"$nb_sample_raw
echo $pkl_folder
echo "PERCO: dir=" $dir ", size=" $size ",nb_sample=" $nb_sample ", cores=" $cores

# --- working in RAW directories

cd $dir
cd "L"$size"-"$nb_sample_raw
pwd

for directory in W*/
do
    if [ ! -d $destdir"/"$pkl_folder\$directory ]; then
        echo "--- making "$destdir"/"$pkl_folder$directory
        mkdir -p $destdir"/"$pkl_folder$directory
        #cd $destdir"/"$pkl_folder$directory
    else
        echo "--- cd "$destdir"/"$pkl_folder$directory
        #cd $destdir"/"$pkl_folder$directory
    fi

    jobfile=MakePickle-$size".sh"
    pyscript=MakePickle-$size".py"

    echo "--- creating "$jobfile $pyscript "in "$destdir"/"$pkl_folder$directory

cat > $destdir"/"$pkl_folder$directory"/"${pyscript} << EOD
#!/usr/bin/python3
import sys
import numpy as np
import os
import pickle as pkl

#print(sys.argv)

filename, file_extension = os.path.splitext(str(sys.argv[1]))
print('>-- creation of file',filename+'.pkl')

new_path=str(sys.argv[2])+filename+'.pkl'

#print(sys.argv[1])
data=np.loadtxt(str(sys.argv[1]))

pickle_file = open(new_path,"wb")
pkl.dump(data, pickle_file)
pickle_file.close()
EOD

cat > $destdir/$pkl_folder$directory"/"${jobfile} << EOD
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3700

module load GCCcore/11.2.0 parallel/20210722
module load Anaconda3

#conda init --all; conda activate

echo "--> working in directory=$dir/$raw_folder/$directory ---"

cd $dir/$raw_folder/$directory
pwd
rm -f raw.lst

for rawfile in \`ls *.raw | head -$nb_sample\`
do
    if [ ! -f \`basename \$rawfile .raw\`.pkl ]; then
	echo \$rawfile "needs .pkl file"
        #echo "file" \`basename \$rawfile .raw\`.pkl  "does not exist" 
	echo \$rawfile >> raw.lst
    fi
done

pwd

sort -R raw.lst | parallel -j$cores -a - python $destdir/$pkl_folder/$directory/$pyscript {} $destdir"/"$pkl_folder$directory

#if [ ! -f $dir/$raw_folder/$directory/raw.lst ]; then
#    echo "all the files are already converted to pkl"
#else
#    echo "the directory already exists"
#    sort -R raw.lst | parallel -j$cores -a - python $pyscript {} $pkl_folder"/"$directory
#fi

pwd

echo "--> finished in directory=$directory"

EOD

    #cat ${jobfile}
    #cat ${pyscript}

    chmod 755 $destdir"/"$pkl_folder$directory"/"${jobfile}
    chmod g+w $destdir"/"$pkl_folder$directory"/"${jobfile}

    echo "--> starting the .sh file"
    #(sbatch -q devel ${jobfile})
    #(sbatch -q taskfarm ${jobfile})
    #(sbatch ${jobfile})
    (${destdir}"/"$pkl_folder$directory"/"${jobfile})
    
done

cd $codedir

$MAKEDISQS $destdir"/"$pkl_folder
