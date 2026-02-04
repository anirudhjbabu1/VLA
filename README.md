# VLA
# Visual Language Action Model
# Under development

This project is with reference to https://github.com/keivalya/mini-vla
It helped me to get a better understanding of the field.

Also Youtube video series by Ilia - https://youtu.be/8dZUOo5xWFw?si=KlFwSWtGUUlT2xPR

I will be improving and developing custom VLA models which are hardware agnostic. To be deployed on any Unmanned vehicles.


Designing VLA

Image → image encoder → image token

Text → text encoder → language token

State → state encoder → state token

Fusion → combine three tokens → fused context

Diffusion head → sample an action conditioned on the fused context

Visual input + LLM = VLM

<img width="1144" height="940" alt="Screenshot from 2026-02-03 11-25-06" src="https://github.com/user-attachments/assets/8610c15b-8b40-4740-8611-3e40c3c23dc1" />

VLM + Action = VLA

<img width="1395" height="1069" alt="Screenshot from 2026-02-03 11-27-37" src="https://github.com/user-attachments/assets/a9fe11bc-ff9f-4019-bbb8-893fb85ade72" />
