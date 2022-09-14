from pickle import FALSE
from typing import Optional

import fire
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from adapters import add_bert_adapters, AdapterConfig, freeze_all_parameters, unfreeze_bert_adapters
from data import DataFrameTextClassificationDataset,SSTDataset,DataIOSST2
from models import AutoModelForSequenceClassificationFinetuner


def train(
    num_epochs: int = 5,
    n_workers: int = 0,#4
    gpus: int = 1,
    precision: int = 32,
    patience: int = 5,
    adapter_size: Optional[int] = 64,
    lr: float = 2e-05,
    model_name: str = 'bert-base-multilingual-cased',
    train_file: str = 'G:/deltatuning/bert_adapter-master/data/train.xls',
    test_file: str = 'G:/deltatuning/bert_adapter-master/data/test.xls',
    output_dir: str = 'G:/deltatuning/bert_adapter-master/data/output_dir'
):
    torch.random.manual_seed(42)

    train_df = pd.read_csv(train_file,error_bad_lines=False,sep='\t',header=0)
    test_df = pd.read_csv(test_file,error_bad_lines=False,sep='\t',header=0)

    # Take 10% of training data as validation
    val_df = train_df.iloc[int(len(train_df) * 0.9):]
    train_df = train_df.iloc[:int(len(train_df) * 0.9)]

    train_dataset = DataFrameTextClassificationDataset(train_df)
    val_dataset = DataFrameTextClassificationDataset(val_df)
    test_dataset = DataFrameTextClassificationDataset(test_df)
    '''dataset = DataIOSST2({'path': 'G:/deltatuning/bert_adapter-master/data', 'batch_size': 16})
    train_set = SSTDataset(dataset.train_word, dataset.train_label)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=16)
    train_eval_loader = DataLoader(train_set, shuffle=False, batch_size=16)
    dev_set = SSTDataset(dataset.dev_word, dataset.dev_label)
    dev_eval_loader = DataLoader(dev_set, shuffle=False, batch_size=16)

    test_set = SSTDataset(dataset.test_word, dataset.test_label)
    test_eval_loader = DataLoader(test_set, shuffle=False, batch_size=16)'''
    train_loader = DataLoader(train_dataset, num_workers=n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, num_workers=n_workers)

    # Load pre-trained model (weights)
    model = AutoModelForSequenceClassificationFinetuner(model_name,
                                                        n_classes=2,
                                                        lr=lr)
    if adapter_size is not None:
        # Add adapters and freeze all layers
        config = AdapterConfig(
            hidden_size=768, adapter_size=adapter_size,
            adapter_act='relu', adapter_initializer_range=1e-2
        )
        model.model.bert = add_bert_adapters(model.model.bert, config)
        model.model.bert = freeze_all_parameters(model.model.bert)

        # Unfreeze adapters and the classifier head
        model.model.bert = unfreeze_bert_adapters(model.model.bert)
        model.model.classifier.requires_grad = True
    else:
        print("Warning! BERT adapters aren't used because adapter_size wasn't specified.")

    trainer = pl.Trainer(max_epochs=num_epochs,
                         gpus=gpus,
                         auto_select_gpus=gpus > 0,
                         auto_scale_batch_size=True,
                         precision=precision,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=patience)])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    model.save_inference_artifact(output_dir)


if __name__ == '__main__':
    fire.Fire(train)
