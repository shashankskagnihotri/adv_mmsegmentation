#!/bin/bash

epsilon=8
alpha=2.55
name='apgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py pretrained_models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/dag_comparison/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=6
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py pretrained_models/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/dag_comparison/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=1
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=2
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=3
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=4
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=5
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=6
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=7
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=8
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=9
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=10
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=11
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=12
alpha=2.55
name='cospgd'
iterations=3
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir








epsilon=1
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=2
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=3
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=4
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=5
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=6
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=7
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=8
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=9
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=10
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=11
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=12
alpha=2.55
name='cospgd'
iterations=5
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir










epsilon=1
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=2
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=3
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=4
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=5
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=6
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=7
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=8
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=9
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=10
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=11
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=12
alpha=2.55
name='cospgd'
iterations=10
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir







epsilon=1
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=2
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=3
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=4
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=5
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=6
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=7
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=8
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=9
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=10
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=11
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=12
alpha=2.55
name='cospgd'
iterations=20
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir











epsilon=1
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=2
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=3
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=4
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=5
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=6
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=7
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=8
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=9
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=10
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=11
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=12
alpha=2.55
name='cospgd'
iterations=40
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir











epsilon=1
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=2
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

epsilon=3
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=4
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=5
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=6
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=7
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=8
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=9
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=10
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=11
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


epsilon=12
alpha=2.55
name='cospgd'
iterations=100
norm='linf'
python -W ignore tools/test.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py pretrained_models/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth  --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir icml2024_rebutal/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir icml2024_rebutal//attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
