OpenVLA from Stanford
OpenVLA is a open source 7B parameter model trained on 970K episodes of Open X-Embodiment dataset for generalist robotic manipulation tasks.

It consists of three main components:

Vision Encoder: Uses a dual vision encoder approach with DINOv2 (~300M) and SigLIP (~400M) which takes in an image and creates embeds flattened patches. DINOv2 excels at spatial relationships while SigLIP offers strong language alignment properties. To adapt the vision encoders to new tasks like action prediction it is important to unfreeze and train these model layers as well.
Projector: Vision embeddings are mapped into a shared embedding space of LLM using an MLP projector.
LLM: Llama2 7B model takes in an language instruction and is tokenized. The vision embeddings and text tokens together is passed as a sequence to LLM to generate actions such as changes in position, rotation and gripper state which can be directly used as continuous signals to control robots end effector.
OpenVLA Model Architecture using Llama 2 7B with Dual Vision Encoder Dinov2 and SIgLIP to predict 7D Robot Action including position orientation and gripper state

Franka Emika Panda 7-DoF robot arm is used as test bed operating at 5Hz for evaluation and benchmarks. The project also added support for parameter efficient fine-tuning techniques like LoRA and experimental results shows these PEFT models perform on par with the original model.

Interestingly the authors suggest that we can train VLAs just like LLM , via next token prediction with cross entropy loss. All we need is only 255 action tokens to represent entire action space of a robot given the visual observation and language instruction.

Experimental studies shows that OpenVLA outperforms RT-2-X(55B) despite having 7 times fewer parameters. However OpenVLA natively doesn’t perform well on Out of Distribution (unseen) dataset compared to RT-2 as it wasn’t trained on web data unlike RT-2. Therefore fine-tuning on unseen data distribution helps the model to quickly adapt to the novel episodes.

Code Walkthrough of OpenVLA

``` !git clone https://github.com/openvla/openvla.git
%cd openvla
!pip install -e .
 
!pip install packaging ninja
!ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
 
!pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
```
```
# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
 
import torch
 
# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")
 
# Grab image input & format prompt
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
 
# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
 
# Execute...
robot.act(action, ...)

```
