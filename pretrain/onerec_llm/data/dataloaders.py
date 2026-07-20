
from torchdata.stateful_dataloader import StatefulDataLoader
from onerec_llm.data.qwen3_dataset import Qwen3ChatCompletionParquetDataset

def get_chat_completion_parquet_dataloader(sources: str,
                                          max_length,
                                          base_model_dir,
                                          num_epochs=1,
                                          shuffle_seed=1024,
                                          num_workers=8,
                                          datasource_config={},
                                          **kwargs):
    model_type = kwargs.get('model_class','Qwen3ForCausalLM')
    ModelDataset = {'Qwen3ForCausalLM': Qwen3ChatCompletionParquetDataset}
    num_readers = kwargs.get("num_readers", 1)
    shuffle_window = kwargs.get("shuffle_window", 0)

    def input_creator():
        return ModelDataset[model_type](
            sources = sources,
            num_workers = num_workers,
            num_epochs = num_epochs,
            shuffle_seed = shuffle_seed,
            max_length = max_length,
            base_model_dir=base_model_dir,
            datasource_config=datasource_config,
            num_readers=num_readers,
            shuffle_window=shuffle_window,
            **kwargs
            )

    dataset = input_creator()
    dataloader = StatefulDataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],
    )
    return dataloader


def get_dataloader(name: str, **kwargs):
    if name == "chat_completion_parquet":
        return get_chat_completion_parquet_dataloader(
            **kwargs
        )
    else:
        raise NotImplementedError("Unsupported dataloader.")


