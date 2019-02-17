#!/bin/bash

REL=`realpath $BASH_SOURCE`
BASEPATH=`dirname $REL`
EXTRA_PYTHONPATH=${BASEPATH}/src
echo "Adding $EXTRA_PYTHONPATH in python path"
export PYTHONPATH=${EXTRA_PYTHONPATH}:${PYTHONPATH}
