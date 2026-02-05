Step 1: Environment Setup
```
conda create -n openvla python=3.10 -y
conda activate openvla
pip install torch torchvision torchaudio
pip install -e .  # Install OpenVLA and dependencies
pip install "flash-attn==2.5.5" --no-build-isolation
```

Step 2: Run LoRA Fine-Tuning
```
python scripts/finetune.py \
  --model_family "openvla" \
  --pretrained_checkpoint "openvla/openvla-7b" \
  --dataset_name "your_custom_rlds_dataset" \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "my-vla-project"
```
