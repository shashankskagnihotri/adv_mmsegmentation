#!/bin/bash

epsilon=8
alpha=2.55
name='apgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py pretrained_models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir



epsilon=8
alpha=2.55
name='apgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py pretrained_models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir



epsilon=8
alpha=2.55
name='apgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py pretrained_models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir



epsilon=8
alpha=2.55
name='apgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py pretrained_models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir



epsilon=8
alpha=2.55
name='apgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py pretrained_models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir



epsilon=8
alpha=2.55
name='apgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py pretrained_models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal_apgd_corrected/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
