#!/bin/bash

trainingsize=${1:-0}
testsize=${2:-0} 

#zip -rqmT data.zip *

mkdir -p training_set
mkdir -p test_set

unzip data.zip -d training_set/
unzip data.zip -d test_set/

# delete in the test set the training data
cd test_set
pwd

for dir in W*/
do
    echo $dir; cd $dir
    ls -1 `Evec*.raw` | tail -$trainingsize | xargs rm -f
    cd ..
done
cd ..

# delete in the training set the test data
cd training_set
pwd

for dir in W*/
do
    echo $dir; cd $dir
    ls -1 `Evec*.raw` | tail -$testsize | xargs rm -f
    cd ..
done
cd ..




