#!/bin/bash

epsilon=8
alpha=2.55
name='pgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
