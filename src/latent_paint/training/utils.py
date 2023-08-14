
'''
ming
'''
from src.utils import make_path
from src.stable_diffusion import StableDiffusion

def init_mesh_model(latent_mode) -> bool:
    '''
    create the model, latent-nerf, including mesh and render
    '''
    if latent_mode == 'texture-mesh':
        mode = True
     
    elif latent_mode == 'texture-rgb-mesh':
        mode = False
    else:
        raise NotImplementedError(f'--backbone {latent_mode} is not implemented!')
    return mode

def init_diffusion(diffusion_model) -> StableDiffusion:
        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model 


def create_paths(base_path):
    return (
        make_path(base_path),
        make_path(base_path / 'checkpoints'),
        make_path(base_path / 'vis' / 'train'),
        make_path(base_path / 'vis' / 'eval'),
        make_path(base_path / 'results')
    )