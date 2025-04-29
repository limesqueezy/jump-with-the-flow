import torch
import lightning as L
import numpy as np
import torch.autograd as autograd
from jump_wtf.losses.koopman_loss import loss
from jump_wtf.utils.sampling import sample_efficient
from torchmetrics.image.fid import FrechetInceptionDistance
from jump_wtf.utils.fid import make_fid_metric, compute_real_stats
import debugpy

class Model(L.LightningModule):
    def __init__(self, dynamics, autoencoder, koopman, 
                 loss_function, autoencoder_lr, lie_lr, 
                 lr_scheduler, decode_predict_bool=True, 
                 vae_loss_bool=False, koop_reg_bool=False, energy_bool=False, 
                 potential_function=None,  
                 gamma=None, delta_t = 0.01, multistep=False, period=10, time_bool=False, plot_every=50, num_iter=100, cfm_model=None, warmup_step=1000, weight_decay = 0.0, fid_interval=500, fid_real_stats_path="assets/fid_stats/mnist/fid_stats_mnist.pt"):
        super().__init__()
        self.save_hyperparameters(ignore=["autoencoder", "koopman", "loss_function", "dynamics", "potential_function", "cfm_model"])

        #self.save_hyperparameters()  # This helps with checkpointing
        self.autoencoder = autoencoder 
        self.koopman = koopman
        self.warmup_step = warmup_step
        self.loss_function = loss_function
        self.lr_scheduler = lr_scheduler
        self.automatic_optimization = False 
        self.dynamics = dynamics
        self.autoencoder_lr = autoencoder_lr
        self.lie_lr = lie_lr
        self.count = 0 
        self.gamma = gamma
        self.vae_loss = vae_loss_bool
        self.koop_reg  = koop_reg_bool
        self.decode_predict = decode_predict_bool
        self.delta_t = delta_t 
        self.energy_bool = energy_bool 
        self.potential_function = potential_function
        self.multistep = multistep 
        self.period = period
        self.time = None
        self.time_bool = time_bool
        self.plot_every = plot_every 
        self.num_iter = num_iter
        self.cfm_model = cfm_model
        self.weight_decay = weight_decay
        
        self.multistep = multistep

        self.fid_interval = fid_interval

        self.fid_train = make_fid_metric(fid_real_stats_path)
        self.fid_val   = make_fid_metric(fid_real_stats_path)

    def training_step(self, batch, batch_idx):
        
        tensorboard = self.logger.experiment
        optimiser_autoencoder, optimiser_lie = self.optimizers()
        autoencoder_scheduler, lie_scheduler = self.lr_schedulers()
        tensor2d_batch_x, tensor2d_batch_x_next, targets, delta_t = batch

        if (self.global_step == 0) or (self.global_step%self.plot_every==0):
            
            if self.multistep == True:
                self.compute_multistep = True
            
            else: 
                self.compute_multistep = False
            
            if self.time_bool:
                self.time = np.linspace(0,1,self.num_iter)[np.random.randint(0,self.num_iter)]
            #    im, min_error, max_error, total_log_error = plot_sim(self, self.dynamics, self.time, self.time_bool, cfm_model=self.cfm_model)
            #else: 
            #    im, min_error, max_error, total_log_error = plot_sim(self, self.dynamics, time=None, time_dep=self.time_bool)
            sample = sample_efficient(self, t_max=1, n_iter=100)
            sample = sample.clamp(-1, 1)
            tensorboard.add_image("sample", sample, self.global_step, dataformats="NCHW")
        
            #Log relevant info
            #self.log("total_log_error", total_log_error, prog_bar=True)
            #self.log("min_error", min_error, prog_bar=True)
            #self.log("max_error", max_error, prog_bar=True)
            
            #im = torchvision.transforms.functional.pil_to_tensor(im)
            #tensorboard.add_image('images', im, self.global_step)
            #lie_module = (self.koopman(torch.eye(output_dim).to("cuda"))).clone()
            #lie_module = lie_module.detach().cpu()
            #eig_lie, _ = np.linalg.eig(lie_module)
            
            #plt.scatter(np.real(eig_lie), np.imag(eig_lie))
            #plt.savefig("/home/turan/koopman/plots/eig_lie.png")
            #plt.close()    

            #koopman = torch.matrix_exp(lie_module*self.delta_t)
            #eig_lie_im = Image.open("/home/turan/koopman/plots/eig_lie.png")
            #eig_lie_im = torchvision.transforms.functional.pil_to_tensor(eig_lie_im)
            #tensorboard.add_image("eig_lie", eig_lie_im, self.global_step)
            #tensorboard.add_image('lie_module', torch.Tensor(lie_module).unsqueeze(0), self.global_step)
            #tensorboard.add_image("koopman", koopman.unsqueeze(0), self.global_step)
            #tensorboard.add_image("koop_unitarity", torch.matmul(torch.conj(koopman).T, koopman).unsqueeze(0), self.global_step)
            #os.remove("/home/turan/koopman/plots/eig_lie.png")

            #eig_koop, _ = np.linalg.eig(koopman)
            #plt.scatter(np.real(eig_koop), np.imag(eig_koop))
            #plt.savefig("/home/turan/koopman/plots/eig_koop.png")
            #plt.close()    

            
            #eig_koop_im = Image.open("/home/turan/koopman/plots/eig_koop.png")
            #eig_koop_im = torchvision.transforms.functional.pil_to_tensor(eig_koop_im)
            #tensorboard.add_image("eig_koop", eig_koop_im, self.global_step)


            #outputs = self.autoencoder.encoder(batch[0])
            #counts, bins = np.histogram(outputs.detach().flatten().to("cpu").numpy(), bins=50, density=True)
            #plt.stairs(counts, bins)
            #plt.savefig('/home/turan/koopman/plots/hist.png')
            #plt.close()
            #hist = Image.open("/home/turan/koopman/plots/hist.png")
            #hist = torchvision.transforms.functional.pil_to_tensor(hist)
            #tensorboard.add_image("hist", hist, self.global_step)
            #os.remove("/home/turan/koopman/plots/hist.png")
        (encoded, jvp) = \
        autograd.functional.jvp(self.autoencoder.encoder,
                                tensor2d_batch_x,
                                tensor2d_batch_x_next,
                                create_graph=True)
        reconstructed = self.autoencoder.decoder(encoded)   # d(g)
        lie_g_next = self.koopman(encoded)                  # L g
        predicted = self.autoencoder.decoder(lie_g_next)    # d(Lg)
        tensor_loss, reconstructed_loss, grad_phase, decode_predict, vae_loss, koopman_reg, energy_loss, multistep,  sample_loss, sample_loss_phase = loss (
                self, 
                tensor2d_x=tensor2d_batch_x,
                tensor2d_x_next=tensor2d_batch_x_next,
                tensor2d_decoded_x=reconstructed,
                tensor2d_observable=encoded,
                tensor2d_lie_observable_next=lie_g_next,
                tensor2d_predict_x_next=predicted,
                tensor2d_jvp=jvp,targets=targets, time_to_target=delta_t, loss_function=self.loss_function, 
                decode_predict_bool = self.decode_predict, vae_loss_bool=self.vae_loss, 
                delta_t=self.delta_t, koop_reg_bool=self.koop_reg,
                energy_bool=self.energy_bool, 
                potential_function=self.potential_function, multistep_loss_bool=self.compute_multistep, period=self.period, 
                time_bool=self.time_bool, n_iter=self.num_iter, cfm_model=self.cfm_model
            )
        
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.koopman.parameters(), max_norm=0.5)
            
        if self.lr_scheduler == "ReduceLROnPlateau":
            autoencoder_scheduler.step(tensor_loss)
            lie_scheduler.step(tensor_loss)
                    
        else:
            autoencoder_scheduler.step()
            lie_scheduler.step()
            
        self.compute_multistep = False

        if self.global_step%self.plot_every == 0:
            self.log("multistep_loss",multistep, prog_bar=True)
        
        self.log("reconstructed", reconstructed_loss, prog_bar=True)
        self.log("grad_phase", grad_phase, prog_bar=True)
        self.log("decode_predict", decode_predict, prog_bar=True)
        self.log("vae_loss", vae_loss, prog_bar=True)
        self.log("koop_reg", koopman_reg, prog_bar=True)
        self.log("energy_loss", energy_loss, prog_bar=True)
        self.log("sample_loss", sample_loss, prog_bar=True)
        self.log("sample_loss_phase", sample_loss_phase, prog_bar=True)
        self.log("total_loss", tensor_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        self.manual_backward(tensor_loss)
        optimiser_autoencoder.step()
        optimiser_lie.step()
        
        optimiser_lie.zero_grad()
        optimiser_autoencoder.zero_grad()
        self.log("koopman_lr", optimiser_lie.param_groups[0]['lr'])
        self.log("autoencoder_lr", optimiser_autoencoder.param_groups[0]['lr'])

        # per‑step: for checkpointing
        self.log("train_loss_step", tensor_loss,
                on_step=True,  on_epoch=False, prog_bar=False)

        # epoch‑average: for TensorBoard
        self.log("train_loss", tensor_loss,
                on_step=False, on_epoch=True, prog_bar=True)
        
        self.count += 1
        return tensor_loss

    def configure_optimizers(self):
        learning_rate_autoencoder = self.autoencoder_lr
        learning_rate_lie = self.lie_lr 

        optimiser_autoencoder = torch.optim.Adam(self.autoencoder.parameters(),
                                         lr=learning_rate_autoencoder,
                                         weight_decay=self.weight_decay)
        optimiser_lie = torch.optim.Adam(self.koopman.parameters(),
                                 lr=learning_rate_lie,
                                 weight_decay=self.weight_decay)
        
        if self.lr_scheduler ==  "ExponentialLR":
            lie_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser_lie, gamma=self.gamma)
            autoencoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser_autoencoder, gamma=self.gamma)
            return [optimiser_autoencoder, optimiser_lie], [autoencoder_scheduler, lie_scheduler]

        elif self.lr_scheduler == "CosineAnnealingWarmRestarts":
        
            lie_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser_lie, T_0=4500, T_mult=2)
            autoencoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser_autoencoder, T_0=4500, T_mult=2)
            return [optimiser_autoencoder, optimiser_lie], [autoencoder_scheduler, lie_scheduler]
        
        elif self.lr_scheduler == "ReduceLROnPlateau":
            lie_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser_lie, patience=100, threshold=1e-3, factor=0.995)
            autoencoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser_autoencoder, patience=100, threshold=1e-3, factor=0.995)
            return [optimiser_autoencoder, optimiser_lie], [autoencoder_scheduler, lie_scheduler]