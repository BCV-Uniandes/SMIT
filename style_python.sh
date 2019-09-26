#!/bin/bash
for i in *py misc/*py models/*py datasets/*.py;
do
    echo "Formating $i"
    yapf -i $i
    autopep8 --in-place -a -a --recursive $i
    yapf -i $i; flake8 $i
    echo "---"
done
