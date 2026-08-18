from types import SimpleNamespace

from gem.callbacks.prog_bar import ProgressReporter


def test_progress_reporter_snapshots_partial_epoch_total() -> None:
    trainer = SimpleNamespace(
        is_global_zero=True,
        current_epoch=1,
        global_step=51,
        max_epochs=-1,
        max_steps=100,
        num_training_batches=51,
    )
    reporter = ProgressReporter(exp_name="test", data_name="test")
    reporter.setup(trainer, SimpleNamespace(), "fit")
    reporter.on_train_epoch_start(trainer)

    assert reporter.train_epoch_total == 49
    trainer.global_step = 75
    assert reporter.total_train_batches == 25
    assert reporter.train_epoch_total == 49
