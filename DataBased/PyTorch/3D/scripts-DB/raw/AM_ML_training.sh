#!/bin/bash

dir=${1:-../data}
seed=${2:-12345684}
py=${3=Train-Pytorch-class_cor.py}
flag=${4:-0}
size=${5:-50}
lr=${6:-0.001}
batch_size=${7:-32}
size_samp=${8:-5000}
psi_type=${9:-squared}
classes=${10:-'15.0,15.25,15.5,15.75,16.0,16.2,16.3,16.4,16.5,16.6,16.7,16.8,17.0,17.25,17.5,17.75,18.0'}


validation_split=0.1
num_epochs=50
codedir=`pwd`

echo "PERCO: dir=" $dir ",seed:"$seed ",py="$py" ,size:"$size ", size_samp:"$size_samp ", validation_split:"$validation_split ", batch_size:"$batch_size ", num_epochs:"$num_epochs ", flag:"$flag ", size_samp:"$size_samp ", psi_type:"$psi_type

cd `dirname $dir`

mkdir -p `basename $dir`
cd `basename $dir`

mkdir -p $seed
cd $seed
	
jobfile="training-"$seed".sh"

#echo $jobfile

	cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1



module restore TorchGPU_1_12_1
#conda init --all; conda activate

pwd
echo "--- working in directory=$seed"

srun python $codedir/$py $seed $size $size_samp $validation_split $batch_size $num_epochs $flag $lr $psi_type $classes


#echo "--- finished in directory=  $seed"
EOD


cat ${jobfile}
chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
#(sbatch ${jobfile})
(sbatch ${jobfile})
#(./${jobfile})
cd ..


cd $codedir

