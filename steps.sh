#!/bin/bash

morphology='../morphologies/DH070313-.Edit.scaled.swc'
cell_type='rs'
for simplify in 1 2 5 10 ; do
    nohup python steps.py -a -0.3 0.3 0.05 -a 0.4 1.5 0.1 -s $simplify -t $cell_type $morphology  > logs/steps_$simplify.log &
    sleep 5
done

