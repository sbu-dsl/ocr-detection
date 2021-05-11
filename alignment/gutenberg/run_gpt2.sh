#!/bin/bash

for value in {0..5}
do
	python run_gpt2.py $value &
done

