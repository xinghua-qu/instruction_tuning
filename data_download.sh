#!/bin/bash
# cd /mnt/bn/multimodal-moderation-xh/Data/ast_datasets/common_voice_6.1/
# wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-6.1-2020-12-11/en.tar.gz

# cd /mnt/bn/multimodal-moderation-xh/Data/ast_datasets/
echo "We use COVOST-2 for AST task, in which corpus-4 version is used"
DIR="./common_voice_4"
if [-d "$DIR"]; then
    echo "folder exist"
else
    mkdir common_voice_4
fi


cd common_voice_4
file="en.tar.gz"
if [-f "$file"]; then
    tar xvzf en.tar.gz
else
    wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz 
    tar xvzf en.tar.gz
fi