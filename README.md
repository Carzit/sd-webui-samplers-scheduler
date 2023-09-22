# sd-webui-samplers-scheduler

## Introduce
A samplers scheduler which can apply different sampler in diffrent generation steps,  

which I hope will be benefit to achieve a balance between generation speed and image quality.

## How to use
SD Version <= 1.5.2

This repository is a extension for sd webui. Just use it as a sd extension!😉  

Choose the Sampler Seniorious to enable the samplers scheduler.

![](https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/example0.png)


This samplers scheduler provides 8 sampler units (Sampler1-8). 

You can choose what kind of sampler used in each unit(choose None to unable), and the inference steps for each unit.  

The image generation process will follow the configurations of these 8 units in sequence. 


![](https://github.com/Carzit/sd-webui-samplers-scheduler/blob/main/images/example1.png)

## More
The idea of this extension was inspired by Seniorious, A Carillon composed of different talismans.  

Different talismans correspond in sequence to make Seniorious a powerful weapon, and so do the samplers in the samplers scheduler.  

In the end, many thanks to Chtholly Nota Seniorious, the happiest girl in the world.
