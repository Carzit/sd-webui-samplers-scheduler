# sd-webui-samplers-scheduler Seniorious

## Introduction
A samplers scheduler which can apply different sampler in diffrent generation steps.  

I hope it will be helpful to achieve a balance between generation speed and image quality.

My paper on arXiv: [Sampler Scheduler for Diffusion Models](https://arxiv.org/abs/2311.06845)  

## WebUI Version
Now compatible with all.  

The script `scripts/load_Seniorious.py` will load corresponding script according to your SD WebUI version automatically.  
(`Seniorious.py` for version <= 1.5.2, `Seniorious_16.py` for version between 1.6.0 and 1.8.0, `Seniorious_19.py` for version >= 1.9.0)

## How to use
This repository is a extension for sd webui. Just place it in the `extension` folder!😉  

Choose the Sampler `Seniorious`, `Seniorious Karras` or `Seniorious Exponential` to enable the samplers scheduler.  

*`Seniorious` uses nomal noise schedule, `Seniorious Karras` uses the noise schedule recommended in [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) (Karras et al., 2022) and `Seniorious Exponential` uses exponential noise schedule recommended in [Improved Order Analysis and Design of Exponential Integrator for Diffusion Models Sampling](https://arxiv.org/abs/2308.02157)*

<img src="https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/samplers_ui.PNG" width="500px">

**Note that SD WebUI has implemented the separation of samplers and noise schedules since version 1.9.0. Therefore, `Seniorious Karras` and `Seniorious Exponential` will be deprecated since 1.9.0. Just like other samplers in SD WebUI, you can specify the noise schedule used by Seniorious in the `Schedule Type`.**

<img src="https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/samplers_ui_v19.PNG" width="500px">

This samplers scheduler provides 8 sampler units (Sampler 1-8). 

You can choose what kind of sampler used in each unit(choose `None` to unable), and the inference steps for each unit.  

The image generation process will follow the configurations of these 8 units in sequence. 

<img src="https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/scheduler_ui.png" width="500px">

**Attention: The total steps should be equal to the sum of the steps in every unit!**  
Open the `Check` accordion and press the `Check` button to check total steps.   
`Total steps in Seniorious` shows the sum of steps in your Sampler Scheduler Settings and `Total steps required` shows the steps you set in webui. They must be equal.  

<img src="https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/check_ui.png" width="500px">

## Available Samplers
14 kinds of mainstream samplers in [k-diffusion](https://github.com/crowsonkb/k-diffusion) are available:  

- `Euler`
- `Euler a`
- `Heun`
- `Heun++`
- `LMS`
- `DPM2`
- `DPM2 a`
- `DPM++ 2S a`
- `DPM++ SDE`
- `DPM++ 2M`
- `DPM++ 2M SDE`
- `DPM++ 3M SDE`
- `Restart` 
- `LCM`

You can also choose `Skip` to skip certain steps.  

`Heun++` is my [test version](https://github.com/Carzit/sd-webui-sampler-heunpp).  
`Restart` is a new sampler in SD WebUI since version 1.6, using the recommended hyperparameters recommended in [Implements restart sampling in Restart Sampling for Improving Generative Processes](https://arxiv.org/abs/2306.14878).  
`LCM` is a new sampler used in latent consistency model(LCM).  

## FID Result
I calculate the FID Score based on [EDM](https://github.com/NVlabs/edm).

config:  
- seeds=0-49999  
- network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl  
- NFE=30
  
| Sampler[steps] | Type | FID | 
| :-----: | :----: | :----: |
| Euler[30] | Global ODE | 4.20923 | 
| Heun[16] | Global ODE | 8.8082 |
| DPM++2M[30] | Global ODE | 2.06226 |
| DPM2[16] | Global ODE | 1.94304 |
| Euler a[30] | Global SDE | 14.5688 |
| DPM2 a[16] | Global SDE | 3.58586 |
| DPM++ 2S a[16] | Global SDE | 9.74262 |
| DPM++ SDE[16] | Global SDE | 17.807 |
| Heun[10] - DPM2[5] | Sampler Scheduler ODE | 1.86846 |
| DPM2 a[10] - DPM2[5] | Sampler Scheduler SDE+ODE | 1.84535 |

Discretely scheduling different samplers during the sampling process has proven to be effective at a practical level.
  
## Examples

### ODE Samplers
![](https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/example2.png)  

BRISQUE Score: 
| Sampler | BRISQUE |
| :-----:| :----: |
| Euler | 23.3771 |
| DPM++ 2M | 27.5705 |
| DPM++ 2M Karras | 24.8244 |
| Heun | 21.4943 |
| DPM 2 | 22.1520 |
| Seniorious | 21.3698 |
| Seniorious Karras | 21.6955 |  

Shared Parameters: 
- Prompt: best quality, masterpiece, 1girl, solo, standing, sky, portrait, looking down, floating light particles, sunshine, cloud, depth of field, field, wide shot
- Negative prompt: badhandv4, By bad artist -neg, EasyNegative, NegfeetV2, ng_deepnegative_v1_75t, verybadimagenegative_v1.3
- Steps: 30
- CFG scale: 7
- Seed: 2307198650
- Size: 512x768,
- Model: anything-v5-PrtRE  

Sampler Scheduler Parameters:  
| Unit | Sampler | Steps |
| :-----:| :----: | :----: |
| Sampler1 | Heun | 10 steps |
| Sampler2 | DPM++2M | 10 steps |
| Sampler3 | Euler | 10 steps |  

*Seniorious and Seniorious Karras use the same parameters in this example.

### Put ODE and SDE Samplers Together
![](https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/example3.png)  

Sampler Scheduler Parameters:  
| Unit | Sampler | Steps |
| :-----:| :----: | :----: |
| Sampler1 | DPM2 a | 20 steps |
| Sampler2 | DPM2 | 10 steps |

I recommend to use SDE in the early sampling steps and ODE in the later sampling steps to solve the inherent problems previously caused by using either singly.

## More
The idea of this extension was inspired by Seniorious, a Carillon composed of different talismans.  

Different talismans correspond in sequence to make Seniorious a powerful weapon, and so do the samplers in the samplers scheduler.  

In the end, many thanks to Chtholly Nota Seniorious, the happiest girl in the world.
