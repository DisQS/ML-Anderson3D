#!/bin/bash

dir=${1:-../data}
seed=${2:-12345684}
py=${3=Train-Pytorch-class_cor.py}
flag=${4:-0}
size=${5:-50}
size_img=${6:-100}
lr=${7:-0.001}
classes=${8:-'15.0,15.25,15.5,15.75,16.0,16.2,16.3,16.4,16.5,16.6,16.7,16.8,17.0,17.25,17.5,17.75,18.0'}
size_samp=5000
validation_split=0.1
batch_size=32
num_epochs=50

codedir=`pwd`

echo "PERCO: dir=" $dir ",seed:"$seed ",py="$py" ,size:"$size ", size_img:"$size_img ", size_samp:"$size_samp ", validation_split:"$validation_split ", batch_size:"$batch_size ", num_epochs:"$num_epochs ", flag:"$flag

cd $dir

mkdir $seed
cd $seed
	
jobfile="training-"$seed".sh"

#echo $jobfile

	cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su007-rr-gpu

module restore TorchGPU_1_11_0_
#conda init --all; conda activate

pwd
echo "--- working in directory=$seed"

srun python $codedir/$py $seed $size $size_img $size_samp $validation_split $batch_size $num_epochs $flag $lr $classes



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

