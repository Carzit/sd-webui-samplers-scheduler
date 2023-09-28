# sd-webui-samplers-scheduler Seniorious

## Introduce
A samplers scheduler which can apply different sampler in diffrent generation steps,  

I hope it will be helpful to achieve a balance between generation speed and image quality.

## Requirement
sd webui Version <= 1.5.2  

## How to use
This repository is a extension for sd webui. Just use it as a sd extension!😉  

Choose the Sampler `Seniorious` or `Seniorious Karras` to enable the samplers scheduler.  

*`Seniorious` uses nomal noise scheduler and `Seniorious Karras` uses the noise scheduler  
recommended in [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) (Karras et al., 2022)  

![](https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/example0.PNG)


This samplers scheduler provides 8 sampler units (Sampler 1-8). 

You can choose what kind of sampler used in each unit(choose `None` to unable), and the inference steps for each unit.  

The image generation process will follow the configurations of these 8 units in sequence. 

**Attention: The total steps should be equal to the sum of the steps in every unit!**


![](https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/example1.PNG)

## Available Samplers
12 kinds of mainstream samplers are available:  

- `Euler`
- `Euler a`
- `Heun`
- `Heun++` (my improved version)
- `LMS`
- `DPM2`
- `DPM2 a`
- `DPM++ 2S a`
- `DPM++ SDE`
- `DPM++ 2M`
- `DPM++ 2M SDE`
- `DPM++ 3M SDE`

You can also choose `Skip` to skip certain steps.

## Example
![](https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/example2.png)  

BRISQUE Score: 
| Sampler | BRISQUE(lower-better) |
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

*Seniorious and Seniorious Karras use the same parameters in this example

## More
The idea of this extension was inspired by Seniorious, a Carillon composed of different talismans.  

Different talismans correspond in sequence to make Seniorious a powerful weapon, and so do the samplers in the samplers scheduler.  

In the end, many thanks to Chtholly Nota Seniorious, the happiest girl in the world.
