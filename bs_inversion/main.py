from bispectrum_calculation.bispectrum_calc_org import create_gaussian_pulse, calculate_bispectrum_power_spectrum_efficient
from cnn_vocoder import model as vc_model
import torch
from hparams import hparams, hparams_debug_string
import matplotlib.pyplot as plt

if __name__ == "__main__":
    folder_path = 'figures'
    epoch = 1000
    model_path = f'checkpoints/checkpoint_ep{epoch}.pt'
    model = vc_model.CNNBS(
        # n_heads=hparams.n_heads,
        # layer_channels=hparams.layer_channels,
        # pre_conv_channels=hparams.pre_conv_channels,
        # pre_residuals=hparams.pre_residuals,
        # up_residuals=hparams.up_residuals,
        # post_residuals=hparams.post_residuals
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.cuda()

    #x, t, dt, fs = create_gaussian_pulse(0., 0.1, 100, 1)
    x = torch.randn(100)
    x = torch.tensor(x)
    dt = 1
    bs, ps, f = calculate_bispectrum_power_spectrum_efficient(x, dt)
    x = x.cuda()
    bs_real = bs.real.float()
    bs_imag = bs.imag.float()
    bs_channels = torch.stack([bs_real, bs_imag], dim=-1)
    bs_channels = bs_channels.permute(2, 0, 1)
    bs_channels = torch.FloatTensor(bs_channels).cuda()
    _, x_rec = model(bs_channels)
    x_rec=x_rec.squeeze(0)
    loss = x - x_rec
    
    plt.figure()
    plt.xlabel("time [sec]")
    plt.title('1D signal')
    plt.plot(x.detach().cpu().numpy(), label='x')
    plt.plot(x_rec.detach().cpu().numpy(), label='x_rec')
    plt.legend()
    plt.savefig(f'{folder_path}/x_vs_x_rec_ep{epoch}.png')
    plt.close()
