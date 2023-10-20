#!/bin/bash

dir=${1:-../data}
seed=${2:-12345684}
py=${3=Train-Pytorch-class_cor.py}
flag=${4:-0}
size=${5:-50}
lr=${6:-0.001}
batch_size=${7:-128}
my_nclasses=${8:-17}


size_samp=5000
validation_split=0.1

num_epochs=50

codedir=`pwd`

echo "PERCO: dir=" $dir ",seed:"$seed ",py="$py" ,size:"$size ", size_samp:"$size_samp ", validation_split:"$validation_split ", batch_size:"$batch_size ", num_epochs:"$num_epochs ", flag:"$flag

cd $dir

mkdir $seed
cd $seed
	
jobfile="training-"$seed".sh"

#echo $jobfile

	cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700




module restore TorchCPU_1_12_1
#conda init --all; conda activate

pwd
echo "--- working in directory=$seed"

srun python $codedir/$py $seed $size $size_samp $validation_split $batch_size $num_epochs $flag $lr $my_nclasses



#echo "--- finished in directory=  $seed"
EOD


cat ${jobfile}
chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
(sbatch ${jobfile})
#(sbatch ${jobfile})
#(./${jobfile})
cd ..


cd $codedir

