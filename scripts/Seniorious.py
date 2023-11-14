import os
import sys
import functools
import torch
from torch import nn

from modules import shared
import k_diffusion.sampling
from scripts.ksampler import sample_euler,sample_euler_ancestral,sample_heun,sample_heunpp2,sample_lms,sample_dpm_2,sample_dpm_2_ancestral,sample_dpmpp_2s_ancestral,sample_dpmpp_sde,sample_dpmpp_2m,sample_dpmpp_2m_sde,sample_dpmpp_3m_sde,sample_skip

from modules import sd_samplers, sd_samplers_common
import modules.sd_samplers_kdiffusion as K
import modules.scripts as scripts
from modules import shared, script_callbacks
import gradio as gr
import modules.ui
from modules.ui_components import ToolButton, FormRow



MAX_SAMPLER_COUNT=8

# 'DPM fast', 'DPM adaptive',
samplers_list = ['Euler','Euler a', 'Heun', 'Heun++',
                 'LMS',
                 'DPM2','DPM2 a',
                 'DPM++ 2S a','DPM++ SDE',
                 'DPM++ 2M', 'DPM++ 2M SDE', 'DPM++ 3M SDE',
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
                     'Skip':sample_skip,
                     'None':None
                     }



ui_info = [(None, 0) for i in range(MAX_SAMPLER_COUNT)]

class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Samplers Scheduler Seniorious"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        def update_info(sampler, step, i):
            ui_info[i] = (sampler, int(step))
            if sampler == 'None':
                ui_info[i] = (None, 0)

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


        return None

#==================================================================================
#'DPM fast':sample_dpm_fast,
#'DPM adaptive':sample_dpm_adaptive,



def split_sigmas(sigmas, steps):
    result = []
    start = 0
    for num in steps:
        end = start + num
        result.append(sigmas[start:end+1])
        start = end
    return result

def get_samplers_steps():
    result = []
    for i in ui_info:
        if i[0] != None and i[1] != 0:
            result.append(i)
    return result
@torch.no_grad()
def seniorious(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    callback_ = callback
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
class KDiffusionSamplerLocal(K.KDiffusionSampler):
    
    def __init__(self,funcname: str,original_funcname: str,func,sd_model: nn.Module):
        denoiser = k_diffusion.external.CompVisVDenoiser if sd_model.parameterization == "v" else k_diffusion.external.CompVisDenoiser

        self.model_wrap = denoiser(sd_model, quantize=shared.opts.enable_quantization)
        self.funcname = funcname
        self.func = func
        self.extra_params = K.sampler_extra_params.get(original_funcname, [])
        self.model_wrap_cfg = K.CFGDenoiser(self.model_wrap)
        self.sampler_noises = None
        self.stop_at = None
        self.eta = None
        self.config = None
        self.last_latent = None

        self.conditioning_key = sd_model.model.conditioning_key # type: ignore


def add_seniorious():
    original = [ x for x in K.samplers_k_diffusion if x[0] == 'Heun' ][0]
    o_label, o_constructor, o_aliases, o_options = original
    
    label = 'Seniorious'
    funcname = seniorious.__name__
    
    def constructor(model: nn.Module):
        return KDiffusionSamplerLocal(funcname, o_constructor, seniorious, model)
    
    aliases = ['seniorious']
    options = { **o_options }

    data = sd_samplers_common.SamplerData(label, constructor, aliases, options)
    if len([ x for x in sd_samplers.all_samplers if x.name == label ]) == 0:
        sd_samplers.all_samplers.append(data)


def add_seniorious_karras():
    original = [x for x in K.samplers_k_diffusion if x[0] == 'Heun'][0]
    o_label, o_constructor, o_aliases, o_options = original
    o_options = {'scheduler': 'karras'}

    label = 'Seniorious Karras'
    funcname = seniorious.__name__

    def constructor(model: nn.Module):
        return KDiffusionSamplerLocal(funcname, o_constructor, seniorious, model)

    aliases = ['seniorious_karras']
    options = {**o_options}

    data = sd_samplers_common.SamplerData(label, constructor, aliases, options)
    if len([x for x in sd_samplers.all_samplers if x.name == label]) == 0:
        sd_samplers.all_samplers.append(data)

def update_samplers():
    sd_samplers.set_samplers()
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}


def hook(fn):
    
    @functools.wraps(fn)
    def f(*args, **kwargs):
        old_samplers, mode, *rest = args
        
        if mode not in ['txt2img', 'img2img']:
            print(f'unknown mode: {mode}', file=sys.stderr)
            return fn(*args, **kwargs)
        
        update_samplers()
        
        new_samplers = (
            sd_samplers.samplers if mode == 'txt2img' else
            sd_samplers.samplers_for_img2img
        )
        
        return fn(new_samplers, mode, *rest, **kwargs)
    
    return f


# register new sampler
add_seniorious()
add_seniorious_karras()
update_samplers()


# hook Sampler textbox creation
from modules import ui
ui.create_sampler_and_steps_selection = hook(ui.create_sampler_and_steps_selection)
