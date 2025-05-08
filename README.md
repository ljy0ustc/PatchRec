# PatchRec

- *2024.3*: Our [checkpoints](https://huggingface.co/joyliao7777/PatchRec) are released on the huggingface.
- 🔥 *2024.4*: Our paper is accepted by SIGIR'25! Thank all Collaborators! 🎉🎉

##### Preparation

1. Prepare the environment: 

   ```sh
   git clone https://github.com/ljy0ustc/PatchRec.git
   cd PatchRec
   pip install -r requirements.txt
   ```

2. Prepare the pre-trained huggingface model of LLaMA3.2-1B-Instruct (https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

##### Train PatchRec

Please see `train.sh` for training PatchRec with multiple GPUs.

##### Evaluate PatchRec

Please see `test.sh` for evaluating PatchRec with a single GPU.