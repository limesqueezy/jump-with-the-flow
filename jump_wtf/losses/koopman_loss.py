import torch
from tqdm import tqdm
import numpy as np
from jump_wtf.operators.utils import get_koop_continuous_batch

def loss(model, tensor2d_x: torch.Tensor,
         tensor2d_x_next: torch.Tensor,
         tensor2d_decoded_x: torch.Tensor,
         tensor2d_observable: torch.Tensor,
         tensor2d_lie_observable_next: torch.Tensor,
         tensor2d_predict_x_next: torch.Tensor,
         tensor2d_jvp: torch.Tensor, targets,
         time_to_target, loss_function, vae_loss_bool=False,         decode_predict_bool=False,
         koop_reg_bool=False,
         energy_bool = False,
         potential_function = None,
         delta_t=0.01,
         period=10,
         multistep_loss_bool=False, 
         time_bool=False, n_iter=100, 
         grad_phase_weight_factor: float = 0.0,
         cfm_model=None, 
         device='cuda'):
    lie_operator = model.koopman
    autoencoder = model.autoencoder

    loss = loss_function
    output_dim = model.koopman.operator_dim
    # print(time_to_target)
    evolution_operators = get_koop_continuous_batch(
        model,
        output_dim,
        time_to_target.to(device)[0]
    )
    # print(evolution_operators.shape)
    encoded_points = model.autoencoder.encoder(tensor2d_x)
    # print(encoded_points.shape)
    predicted_samples_phase = (encoded_points @ evolution_operators).squeeze(0)
    # print(predicted_samples.shape)
    predicted_samples = model.autoencoder.decoder(predicted_samples_phase)

    sample_loss = loss(targets, predicted_samples)
    sample_loss_phase = loss(
        predicted_samples_phase,
        model.autoencoder.encoder(targets)
    )
    #########################################
    # multistep_loss is undefined, remove?
    if not(multistep_loss_bool):  # model.current_epoch<100:
        multistep = 0
    else:
        if multistep_loss_bool == True:
            t_max = 1  # Number of time steps integration
            n_initial_conditions = len(tensor2d_x)  # Number of initial conditions
            dim_system = 2
            array_t = np.linspace(0, t_max, n_iter)
            solutions = np.zeros((tensor2d_x.shape[0], dim_system, n_iter))

            for i in tqdm(range(tensor2d_x.shape[0])):
                if time_bool:
                    if cfm_model != None:
                        result = solve_ivp(
                            lambda t, x: model.dynamics(cfm_model, t, x),
                            [0, t_max],
                            tensor2d_x[:, :2].cpu().numpy()[i],
                            method="RK45",
                            t_eval=array_t
                        )
                    else:
                        result = solve_ivp(
                            lambda t, x: model.dynamics(t, x),
                            [0, t_max],
                            tensor2d_x[:, :2].cpu().numpy()[i],
                            method="RK45",
                            t_eval=array_t
                        )
                else:
                    result = solve_ivp(
                        lambda _t, x: model.dynamics(x),
                        [0, t_max],
                        tensor2d_x.cpu().numpy()[i],
                        method="RK45",
                        t_eval=array_t
                    )
                solutions[i, :] = result.y

            # solutions = torch.Tensor(solutions).to("cuda")
            multistep = multistep_loss(model, tensor2d_x, solutions, period, time_bool)

    # reconstruction loss
    for p in lie_operator.parameters():
        p.requires_grad = True
    # for p in lie_operator.parameters():
    #     p.requires_grad = False

    for p in autoencoder.encoder.parameters():
        p.requires_grad = True
    # for p in autoencoder.encoder.parameters():
    #     p.requires_grad = False

    reconstructed = loss(tensor2d_decoded_x, tensor2d_x)
    # the decoder doesn't intervene here

    for p in autoencoder.decoder.parameters():
        p.requires_grad = False
    # for p in autoencoder.decoder.parameters():
    #     p.requires_grad = True

    grad_phase = loss(tensor2d_lie_observable_next, tensor2d_jvp)
    # predict only need the decoder so fix the encoder
    # for p in autoencoder.encoder.parameters():
    #     p.requires_grad = False

    decode_predict = loss(tensor2d_predict_x_next, tensor2d_x_next)
    # for p in autoencoder.encoder.parameters():
    #     p.requires_grad = True

    for p in autoencoder.decoder.parameters():
        p.requires_grad = False

    for p in lie_operator.parameters():
        p.requires_grad = False

    if vae_loss_bool:
        vae_loss = (
            torch.abs(torch.mean(tensor2d_observable))
            + torch.abs(torch.std(tensor2d_observable) - 1)
        )
    else:
        vae_loss = 0

    for p in autoencoder.decoder.parameters():
        p.requires_grad = True

    for p in lie_operator.parameters():
        p.requires_grad = True
    # for p in autoencoder.decoder.parameters():
    #     p.requires_grad = False
    # for p in autoencoder.encoder.parameters():
    #     p.requires_grad = False

    lie_module = model.koopman(torch.eye(output_dim).to("cuda"))
    if koop_reg_bool:
        koopman = torch.matrix_exp(lie_module * delta_t)
        koopman_reg = torch.sum(
            torch.abs(
                torch.matmul(torch.conj(koopman.T), koopman)
                - torch.eye(koopman.shape[0]).to(device)
            )
        )
    else:
        koopman_reg = 0
    # for p in autoencoder.decoder.parameters():
    #     p.requires_grad = True
    # for p in autoencoder.encoder.parameters():
    #     p.requires_grad = True

    if not(decode_predict_bool):
        decode_predict = 0

    output_dim = lie_operator.operator_dim

    # Setting energy_bool to true would yield an error energy_loss undef
    if energy_bool:
        energy_loss_term = energy_loss(
            model, tensor2d_x, 0.01, output_dim, potential_function
        )
        orig_loss = (
            0.2 * reconstructed
            + 0.6 * grad_phase
            + 0.2 * decode_predict
            + 0.1 * vae_loss
            + 0.1 * koopman_reg
            + 0.6 * energy_loss_term
            + 0.01 * multistep
        )
    else:
        energy_loss_term = 0
        orig_loss = (
            0.2 * reconstructed
            + grad_phase
            + 0.1 * vae_loss
            # + 0.5 * sample_loss
            # + 0.5 * sample_loss_phase
            + sample_loss_phase
            + 0.1 * koopman_reg
            + 0.01 * multistep
        )
    # for p in lie_operator.parameters():
    #     p.requires_grad = True
    # for p in autoencoder.encoder.parameters():
    #     p.requires_grad = True

    total_loss = grad_phase_weight_factor * grad_phase + (1.0 - grad_phase_weight_factor) * orig_loss

    # TODO: Implement loss
    return (
        total_loss,
        0.2 * reconstructed,
        grad_phase,
        0.2 * decode_predict,
        0.1 * vae_loss,
        0.1 * koopman_reg,
        0.01 * energy_loss_term,
        0.01 * multistep,
        0.5 * sample_loss,
        0.5 * sample_loss_phase,
    )