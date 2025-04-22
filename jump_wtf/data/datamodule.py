from __future__ import annotations
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import lightning as L
from .samplers import TimeGroupedSampler, ListSampler



class DynamicsDataModule(L.LightningDataModule):
    """Builds the (t, x, v, x₁, Δt) tuples once and serves them epoch‑after‑epoch."""
    def __init__(
        self,
        traj: torch.Tensor,
        dynamics,
        batch_size: int = 64,
        t_grid: int = 100,
        val_frac: float = 0.2,
        device = "cuda"
    ):
        super().__init__()
        self.traj       = traj          # (T, B, C, H, W)
        self.dynamics   = dynamics.to(device)
        self.batch_size = batch_size
        self.t_grid     = torch.linspace(0, 1, t_grid)
        self.val_frac   = val_frac
        self.device     = device

    def setup(self, stage=None):
        if not hasattr(self, "full_ds"):  # Ensure dataset is created only once
            
            matrix_x0 = []
            matrix_system_derivative_data = []
            matrix_targets = []
            matrix_delta_t = []

            for i in range(len(self.traj)):
                t = self.t_grid[i] #take the corresponding timesteps from 0 to 1
                t_val = self.t_grid[i]
                
                #Now we get the targets corresponding to the paths we're on
                x1 = self.traj[-1].to(self.device) #the numbers it samples 
                x = self.traj[i].to(self.device)  # Get the 1000 images corresponding to this timestep
               
                # Shuffle the 2000 points (rows) at this time step (for time-ordering, more efficient with Koopman operator)
                #perm = torch.randperm(x.size(0))
                #x = x[perm]

                t = torch.ones((x.shape[0], 1), device=x.device) * t_val #Extend the time so that we can concatenate to get (t,x) that goes in encoder
                t = t.to(self.device)  
                
                #We compute the time required from where we are to get to the target, so we can compute the appropriate evolution operator
                delta_t = 1-t #we keep track of the time to elapse, so that we can regularize sampling
                delta_t = delta_t.to(self.device)
                
                
                with torch.no_grad():
                    #print(f"t device: {t.device}")
                    #print(f"x device: {x.device}")
                    #print(f"dynamics device: {next(self.dynamics.parameters()).device}")
                    #print(self.dynamics.device)
                    dx = self.dynamics(t[:], x)  # get the fixed v(t,x) vector field. Important for the Lg-grad_g*v loss term

                matrix_system_derivative_data.append(dx)
                
                t_one = torch.ones((x.shape[0], 1), device=x.device)
                x1 = x1.reshape((len(x1),-1))
                
                
                targets = torch.hstack((t_one,x1))

                matrix_targets.append(targets)
                
                matrix_delta_t.append(delta_t)
                x = x.reshape((len(x), -1))
                inputs = torch.hstack((t, x)) #Make the inputs of the encoder
                matrix_x0.append(inputs)

            print(len(matrix_targets))

            #The targets
            matrix_targets = torch.vstack(matrix_targets)

            #Time to target
            matrix_delta_t = torch.vstack(matrix_delta_t)

            matrix_x0 = torch.vstack(matrix_x0) #stack to get all (t,x) point for all 1000 trajectories
            matrix_system_derivative_data = torch.vstack(matrix_system_derivative_data) #get all the vector field at these corresponding points
            

            # Train-test split
            #matrix_x_data_train, matrix_x_data_test, matrix_x_next_data_train, matrix_x_next_data_test = train_test_split(
            #    matrix_x0, matrix_system_derivative_data, test_size=0.2, random_state=42
            #)

            # Create datasets
            print("matrix_x0:", matrix_x0.shape)
            print("matrix_system_derivative_data:", matrix_system_derivative_data.shape)
            print("matrix_targets:", matrix_targets.shape)
            print("matrix_delta_t:", matrix_delta_t.shape)

            #Now we feed x, f(x), x1 and time_to_target
            self.full_ds = TensorDataset(matrix_x0.float().to("cpu"), matrix_system_derivative_data.float().to("cpu"), matrix_targets.float().to("cpu"),matrix_delta_t.float().to("cpu"))
            #self.test_dataset = TensorDataset(matrix_x_data_test.float(), matrix_x_next_data_test.float())

            # split indices into train / val
            n = len(self.full_ds)
            n_val = int(self.val_frac * n)
            n_train = n - n_val
            train_range = set(range(0, n_train))
            val_range   = set(range(n_train, n))

            # build time‑grouped ordering over full dataset
            full_sampler = TimeGroupedSampler(
                time_steps=len(self.t_grid),
                group_size=self.traj.size(1),  # B for each time slice
            )
            full_order = full_sampler.indices

            # filter & remap into two lists of positions
            self.train_ordered_indices = [i for i in full_order if i in train_range]
            self.val_ordered_indices   = [i for i in full_order if i in val_range]

            # wrap into ListSampler instances
            self.train_sampler = ListSampler(self.train_ordered_indices)
            self.val_sampler   = ListSampler(self.val_ordered_indices)

    def train_dataloader(self):
        # sample from full_ds in time‑grouped, split‑aware order
        return DataLoader(
            self.full_ds,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.full_ds,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
