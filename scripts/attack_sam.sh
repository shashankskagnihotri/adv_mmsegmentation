#!/bin/bash

epsilon=8
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/SAM/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal/SAM/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal/SAM/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir




epsilon=8
alpha=2.55
name='segpgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/SAM/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal/SAM/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal/SAM/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir




epsilon=8
alpha=2.55
name='pgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/SAM/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal/SAM/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal/SAM/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
