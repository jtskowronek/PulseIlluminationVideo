checkpoint_config = dict(interval=2)

log_config = dict(
    interval=100,
)
save_image_config = dict(
    interval=400,
)
optimizer = dict(type='Adam', lr=0.0001)

loss = dict(type='MSELoss')

runner = dict(max_epochs=400)

checkpoints=None
resume=None
