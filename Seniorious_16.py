import os
import sys
import functools
import torch
from torch import nn

import gradio as gr

import k_diffusion.sampling
from scripts.ksampler import sample_euler,sample_euler_ancestral,sample_heun,sample_heunpp2,sample_lms,sample_dpm_2,sample_dpm_2_ancestral,sample_dpmpp_2s_ancestral,sample_dpmpp_sde,sample_dpmpp_2m,sample_dpmpp_2m_sde,sample_dpmpp_3m_sde,lcm_sampler,restart_sampler,sample_skip

from modules import sd_samplers, sd_samplers_common
import modules.sd_samplers_kdiffusion as K
import modules.scripts as scripts
from modules import processing
from modules import shared, script_callbacks
import modules.ui
from modules.ui_components import ToolButton, FormRow



MAX_SAMPLER_COUNT=8

ui_info = [(None, 0) for i in range(MAX_SAMPLER_COUNT)]

samplers_list = ['Euler','Euler a', 'Heun', 'Heun++',
                 'LMS',
                 'DPM2','DPM2 a',
                 'DPM++ 2S a','DPM++ SDE',
                 'DPM++ 2M', 'DPM++ 2M SDE', 'DPM++ 3M SDE',
                 'LCM', 'Restart',
                 'Skip',
                 'None']

name2sampler_func = {'Euler':sample_euler,
                     'Euler a':sample_euler_ancestral,
                     'Heun':sample_heun,
                     'Heun++':sample_heunpp2,
                     'LMS': sample_lms,
                     'DPM2':sample_dpm_2,
                     'DPM2 a':sample_dpm_2_ancestral,
                     'DPM++ 2S a':sample_dpmpp_2s_ancestral,
                     'DPM++ SDE':sample_dpmpp_sde,
                     'DPM++ 2M':sample_dpmpp_2m,
                     'DPM++ 2M SDE':sample_dpmpp_2m_sde,
                     'DPM++ 3M SDE':sample_dpmpp_3m_sde,
                     'LCM':lcm_sampler,
                     'Restart':restart_sampler,
                     'Skip':sample_skip,
                     'None':None
                     }

#==================================================================================
# Create UI

class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Samplers Scheduler Seniorious"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        if kwargs.get("elem_id") == "txt2img_steps":
            self.t2i_steps = component
        if kwargs.get("elem_id") == "img2img_steps":
            self.i2i_steps = component

    def ui(self, is_img2img):

        def update_info(sampler, step, i):
            ui_info[i] = (sampler, int(step))
            if sampler == 'None':
                ui_info[i] = (None, 0)

        def get_info_total_steps():
            return sum([s[1] for s in ui_info])

        def get_sd_total_steps():
            if is_img2img:
                return self.i2i_steps
            else:
                return self.t2i_steps

        with gr.Group():
            with gr.Accordion("Samplers Scheduler Seniorious", open=False):
                for i in range(MAX_SAMPLER_COUNT):
                    with FormRow(variant="compact"):
                        sampler = gr.Dropdown(samplers_list,
                                              value="None",
                                              label=f'Sampler{i + 1}')
                        step = gr.Slider(minimum=0,
                                         maximum=50,
                                         step=1,
                                         label=f'Steps{i + 1}')
                        with gr.Row(visible=False):
                            index = gr.Slider(value=i,
                                              interactive=False,
                                              visible=False)
                        sampler.change(update_info,
                                       inputs=[sampler, step, index],
                                       outputs=[])
                        step.change(update_info,
                                    inputs=[sampler, step, index],
                                    outputs=[])
                with gr.Accordion("Check", open=False):
                    with FormRow(variant="compact"):
                        seniorious_steps = gr.Textbox(label="Total steps in Seniorious")
                        sd_steps = gr.Textbox(label="Total steps Required")
                        check_btn = gr.Button(value="Check")
                        check_btn.click(get_info_total_steps, inputs=[], outputs=[seniorious_steps])
                        check_btn.click(lambda x:x, inputs=[get_sd_total_steps()], outputs=[sd_steps])
        return None

#==================================================================================
# Sampler Scheduler

def split_sigmas(sigmas, steps):
    print('split_sigmas:',sigmas)
    result = []
    start = 0
    for num in steps:
        end = start + num
        if (not sigmas[start:end+1]) or sigmas[start:end+1] == [0]:
            break
        else:
            result.append(sigmas[start:end + 1])
            start = end
    return result

def get_samplers_steps():
    result = []
    for i in ui_info:
        if i[0] != None and i[1] != 0:
            result.append(i)
    return result
@torch.no_grad()
def seniorious(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    callback_ = callback

    print('Sampler Scheduler Settings:', end=' ')
    print(ui_info)

    samplers_steps = get_samplers_steps()
    samplers_steps = [(name2sampler_func[sampler_step[0]], int(sampler_step[1])) for sampler_step in samplers_steps]
    samplers = [sampler_step[0] for sampler_step in samplers_steps]
    steps = [sampler_step[1] for sampler_step in samplers_steps]
    splitted_sigmas = split_sigmas(sigmas.tolist(), steps)
    x_ = x

    for i in range(len(splitted_sigmas)):
        s = torch.tensor(splitted_sigmas[i], device='cuda:0')
        x_ = samplers[i](model=model, x=x_, sigmas=s, extra_args=extra_args, callback=callback_)

    return x_

#==================================================================================
# register new sampler

class KDiffusionSamplerLocal(K.KDiffusionSampler):

    def __init__(self, funcname, sd_model, options=None):
        self.funcname = funcname
        self.func = seniorious
        self.extra_params = []
        self.model_wrap_cfg = K.CFGDenoiserKDiffusion(self)
        self.model_wrap = self.model_wrap_cfg.inner_model
        self.sampler_noises = None
        self.stop_at = None
        self.config = None
        self.last_latent = None
        self.options = options or {}


        self.eta = 1
        self.eta_option_field = 'eta_ancestral'
        self.eta_infotext_field = 'Eta'
        self.eta_default = 1.0

        self.conditioning_key = sd_model.model.conditioning_key

    def initialize(self, p) -> dict:
        self.p = p
        self.model_wrap_cfg.p = p
        self.model_wrap_cfg.mask = p.mask if hasattr(p, 'mask') else None
        self.model_wrap_cfg.nmask = p.nmask if hasattr(p, 'nmask') else None
        self.model_wrap_cfg.step = 0
        self.model_wrap_cfg.image_cfg_scale = getattr(p, 'image_cfg_scale', None)

        self.s_min_uncond = getattr(p, 's_min_uncond', 0.0)

        self.s_churn = 0.0
        self.s_tmin = 0.0
        self.s_tmax = float('inf')
        self.s_noise = 1.0


        k_diffusion.sampling.torch = sd_samplers_common.TorchHijack(p)

        extra_params_kwargs = {}
        extra_params_kwargs["Sampler Scheduler Config"] = str(ui_info)
        self.p.extra_generation_params["Sampler Scheduler Config"] = str(ui_info)
        return extra_params_kwargs


def add_seniorious():
    label = 'Seniorious'
    funcname = seniorious.__name__
    aliases = ['seniorious']
    options = {}
    new_sampler = [(label, funcname, aliases, options)]

    data = [sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSamplerLocal(funcname, model),aliases, options)
                                 for label, funcname, aliases, options in new_sampler][0]

    sd_samplers.all_samplers.append(data)


def add_seniorious_karras():
    label = 'Seniorious Karras'
    funcname = seniorious.__name__
    aliases = ['seniorious_karras']
    options = {'scheduler': 'karras'}
    new_sampler = [(label, funcname, aliases, options)]

    data = [sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSamplerLocal(funcname, model),aliases, options)
                                 for label, funcname, aliases, options in new_sampler][0]

    sd_samplers.all_samplers.append(data)

def update_samplers():
    sd_samplers.set_samplers()
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}


add_seniorious()
add_seniorious_karras()
update_samplers()

