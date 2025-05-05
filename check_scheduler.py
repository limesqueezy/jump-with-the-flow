import torch
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt

# 1. Dummy model & optimizer
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 2. Scheduler (use the correct ctor args)
warmup_epochs = 10
total_epochs  = 100
scheduler = LinearWarmupCosineAnnealingLR(
    optimizer,
    warmup_epochs=warmup_epochs,
    max_epochs=total_epochs,
    warmup_start_lr=0.0,
    eta_min=0.0
)

# 3. Step & record LRs
lrs = []
for epoch in range(total_epochs):
    optimizer.step()
    scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])

# 4. Plot & save (no plt.show())
plt.plot(lrs)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Linear Warmup + Cosine Annealing")
plt.tight_layout()
plt.savefig("scheduler_plot.png")
