#!/bin/bash

dir=${1:-../data}
dest_dir=${2:-../data2}
size=${3:-10}
nb_sample=${4:-5000}
cores=${5:-1}

MAKEDISQS=/storage/disqs/MakeDisQS.sh

codedir=`pwd`
pkl_folder="L"$size"-"$nb_sample"-pkl"
echo $pkl_folder
echo "PERCO: dir=" $dir ", size=" $size ",nb_sample=" $nb_sample ", cores=" $cores

cd $dir
cd "L"$size"-"$nb_sample
pwd

for directory in W*
do

echo $directory
cd $directory

jobfile=$size-$directory".sh"
pyscript=MakePickle.py

echo $jobfile

cat > ${pyscript} << EOD
#!/usr/bin/python3
import sys
import numpy as np
import os
import pickle as pkl

#print(sys.argv)
print('creation of file',str(sys.argv[2]))
filename, file_extension = os.path.splitext(str(sys.argv[1]))
new_path=str(sys.argv[2])+'/'+filename+'.pkl'
data=np.loadtxt(str(sys.argv[1]))

pickle_file = open(new_path,"wb")
pkl.dump(data, pickle_file)
pickle_file.close()
EOD




cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3700

module load GCCcore/11.2.0 parallel/20210722
module load Anaconda3

#conda init --all; conda activate

pwd
echo "--- working in directory=$directory ---"

rm -f raw.lst

for rawfile in *.raw
do
    if [ ! -d $dest_dir"/"$pkl_folder"/"$directory ]; then
        echo \$rawfile "missing .pkl file"
        echo "directory" $dest_dir"/"$pkl_folder"/"$directory "does not exist"
	echo \$rawfile >> raw.lst
    elif [ ! -f $dest_dir"/"$pkl_folder"/"$directory"/"\`basename \$rawfile .raw\`.pkl ]; then
	echo \$rawfile "missing .pkl file"
        echo "file" $dest_dir"/"$pkl_folder"/"$directory"/"\`basename \$rawfile .raw\`.pkl  "does not exist" 
	echo \$rawfile >> raw.lst
    fi
done
if [ ! -d $dest_dir"/"$pkl_folder"/"$directory ]; then
    mkdir -p $dest_dir"/"$pkl_folder"/"$directory
    echo "the directory does not exist"
    sort -R raw.lst | parallel -j$cores -a - python $pyscript {} $dest_dir"/"$pkl_folder"/"$directory
elif [ ! -f raw.lst ]; then
    echo "all the files are already converted to pkl"
else
    echo "the directory already exist"
    sort -R raw.lst | parallel -j$cores -a - python $pyscript {} $dest_dir"/"$pkl_folder"/"$directory

fi
pwd

echo "--- finished in directory=$directory"

EOD

cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
#(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
(./${jobfile})

cd ..
done

cd $codedir

$MAKEDISQS $destdir
