# Packages

## KeyBERT, KeyLLM

References:
- [KeyBERT Article](https://www.maartengrootendorst.com/blog/keybert/)
- [KeyLLM Article](https://www.maartengrootendorst.com/blog/keyllm/)

<<KeyBERT>> and <<KeyLLM>> are packages to perform unsupervised keyword extraction from text. KeyBERT relies on BERT-based models, and the main idea is to extract n-gram phrases which have high semantic similarity to the overall document embedding. Some additional features are:
- Allow user to specify phrase length for extraction
- Add diversification via MMR to get diverse phrases

KeyLLM taps on LLMs to enhance the keyword extraction. Basically, it creates a prompt to ask an LLM to extract keywords from a document. It integrates with KeyBERT such that we can use KeyBERT to cluster documents, and only run KeyLLM on one document per cluster to save costs. It can also use KeyBERT to suggest candidates and use the LLM to verify.

## Pytorch Lightning

The following pseudo code captures almost everything we need to know about pytorch lightning. Taken from [here](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks).

```python
def fit(self):
    configure_callbacks()

    if local_rank == 0:
        prepare_data()

    setup("fit")
    configure_model()
    configure_optimizers()

    on_fit_start()

    # the sanity check runs here

    on_train_start()
    for epoch in epochs:
        fit_loop()
    on_train_end()

    on_fit_end()
    teardown("fit")


def fit_loop():
    torch.set_grad_enabled(True)

    on_train_epoch_start()

    for batch in train_dataloader():
        on_train_batch_start()

        on_before_batch_transfer()
        transfer_batch_to_device()
        on_after_batch_transfer()

        out = training_step()

        on_before_zero_grad()
        optimizer_zero_grad()

        on_before_backward()
        backward()
        on_after_backward()

        on_before_optimizer_step()
        configure_gradient_clipping()
        optimizer_step()

        on_train_batch_end(out, batch, batch_idx)

        if should_check_val:
            val_loop()

    on_train_epoch_end()


def val_loop():
    on_validation_model_eval()  # calls `model.eval()`
    torch.set_grad_enabled(False)

    on_validation_start()
    on_validation_epoch_start()

    for batch_idx, batch in enumerate(val_dataloader()):
        on_validation_batch_start(batch, batch_idx)

        batch = on_before_batch_transfer(batch)
        batch = transfer_batch_to_device(batch)
        batch = on_after_batch_transfer(batch)

        out = validation_step(batch, batch_idx)

        on_validation_batch_end(out, batch, batch_idx)

    on_validation_epoch_end()
    on_validation_end()

    # set up for train
    on_validation_model_train()  # calls `model.train()`
    torch.set_grad_enabled(True)
```
