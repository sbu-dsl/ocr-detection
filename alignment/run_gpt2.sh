#!/bin/bash

for value in {0..7}
do
	python run_gpt2.py $value &
done

