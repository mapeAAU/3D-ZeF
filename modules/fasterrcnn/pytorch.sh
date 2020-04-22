#!/usr/bin/env bash
# TODO before starting up running your docker command:
# 1. Change "user" to the name of your Ubuntu user name 
# 2. If you have given your dockerimage another tag than "vap", remember to change dockerimage:tag
# 3. Change the value for GpuIndexes if necessary

USER=
PROJECTNAME=#PATH TO REPO
DOCKERIMAGE=pytorch:vap

HOMENAME=$(basename $HOME)
PROJECTPATH=${HOME}/${PROJECTNAME}
mkdir -p "$PROJECTPATH"
# prime permission so user can have full access to files and directory after exiting docker
setfacl -m d:u:${USER}:rwx "$PROJECTPATH"
docker run --gpus device=0 --rm -v ${PROJECTPATH}:/${PROJECTNAME}/
