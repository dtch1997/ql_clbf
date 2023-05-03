# Supervised Losses

We add a supervised loss on the unsafe states 

The supervised loss has the effect of enforcing the barrier condition

## Figures

Compare with and without the supervised loss:

Without:
![](figures/plot_barrier_baseline.png)
![](figures/plot_value_baseline.png)

With:
![](figures/plot_barrier_unsafe.png)
![](figures/plot_value_unsafe.png)

## Videos

See respective `checkpoints/xxxx/rollout.mp4` 

Checkpoint trained with supervised loss: `vgdae80s`
![](checkpoints/vgdae8os/rollout.mp4)

Checkpoint trained without supervised loss: `yp8twvo`
![](checkpoints/yp8tywvo/rollout.mp4)

The checkpoint trained with supervised loss is clearly more stable
