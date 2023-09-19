import sys
import os
import warnings

# Redireciona stderr para um arquivo temporário
temp_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Imports
import os
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from IPython import display
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.classification import Accuracy


if __name__ == '__main__':

    ''' Verificando o Ambiente de Desenvolvimento '''

    # Verifica se uma GPU está disponível e define o dispositivo apropriado

    processing_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define o device (GPU ou CPU)
    device = torch.device(processing_device)
    print(device)

    ''' Preparando os Dados! '''

    # Seed para inicializar o processo randômico com o mesmo padrão
    seed_everything(7)

    # Definindo o caminho onde ficarar os dados!
    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

    # Definimos o tamanho do batch(Lotes) de imagens de acordo com o hardware
    BATCH_SIZE = 256 if torch.cuda.is_available() else 64

    # Número de workers da CPU a ser usado
    NUM_WORKERS = int(os.cpu_count() / 2)

    # Módulo de transformações nos dados de treino (Data Loader de Treino)
    prep_dados_treino = torchvision.transforms.Compose(
        [
            # Padronizando o tamanho das imagens
            torchvision.transforms.RandomCrop(32, padding=4),
            # Aplicando uma rotação em algumas imagens
            torchvision.transforms.RandomHorizontalFlip(),
            # Converter em tensor
            torchvision.transforms.ToTensor(),
            # Normalização dos pixels em uma mesma escala
            cifar10_normalization(),
        ]
    )

    # Módulo de transformações nos dados de teste (teste após o treinamento) e validação (teste durante o treinamento)
    prep_dados_teste = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization()
        ]
    )

    # Módulo para carregar os dados e aplicar os data loaders
    carrega_dados = CIFAR10DataModule(data_dir=PATH_DATASETS,
                                      batch_size=BATCH_SIZE,
                                      num_workers=NUM_WORKERS,
                                      train_transforms=prep_dados_treino,
                                      test_transforms=prep_dados_teste,
                                      val_transforms=prep_dados_teste)

    ''' Criando o modelo com a arquitetura ResNet '''


    # Módulo para carregar um modelo pré-treinado de arquitetura ResNet sem os pesos (queremos somente a arquitetura)
    def carrega_modelo_pretreinado():
        # Carrega o modelo resnet18, sem os pesos pré-treinados
        modelo = torchvision.models.resnet18(weights=None, num_classes=10)
        # Substitui a primeira camada convolucional do modelo
        modelo.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # Substitui a camada de max pooling (agrupamento máximo)
        modelo.maxpool = nn.Identity()
        return modelo


    # Classe com Arquitetura do Modelo
    class ModeloResnet(LightningModule):
        # Método construtor
        def __init__(self, lr=0.05):
            super().__init__()
            self.save_hyperparameters()
            self.model = carrega_modelo_pretreinado()

        # Método Forward
        def forward(self, x):
            out = self.model(x)
            return F.log_softmax(out, dim=1)

        # Método de um passo de treinamento
        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            self.log("train_loss", loss)
            return loss

        # Método de avaliação
        def evaluate(self, batch, stage=None):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            preds = torch.argmax(logits, dim=1)
            accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
            acc = accuracy(preds, y)

            if stage:
                self.log(f"{stage}_loss", loss, prog_bar=True)
                self.log(f"{stage}_acc", acc, prog_bar=True)

        # Método de um passo de validação
        def validation_step(self, batch, batch_idx):
            self.evaluate(batch, "val")

        # Método de um passo de teste
        def test_step(self, batch, batch_idx):
            self.evaluate(batch, "test")

        # Método de configuração do otimizador
        def configure_optimizers(self):
            # Otimização SGD
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.lr,
                                        momentum=0.9,
                                        weight_decay=5e-4)

            # Passos por época
            steps_per_epoch = 45000 // BATCH_SIZE

            # Scheduler
            scheduler_dict = {
                "scheduler": OneCycleLR(optimizer,
                                        0.1,
                                        epochs=self.trainer.max_epochs,
                                        steps_per_epoch=steps_per_epoch),
                "interval": "step",
            }

            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


    # Cria o modelo (Objeto = Instância da Classe)

    modelo_dl = ModeloResnet(lr=0.05)

    # Loop de treinamento
    # Módulo de treinamento

    treinador = Trainer(max_epochs=30,
                        accelerator="auto",
                        devices=1 if torch.cuda.is_available() else None,
                        logger=CSVLogger(save_dir="logs/"),
                        callbacks=[LearningRateMonitor(logging_interval="step"),
                                   TQDMProgressBar(refresh_rate=10)],
                        )

    # Treinamento
    treinador.fit(modelo_dl, carrega_dados)

    # Avaliação do Modelo
    treinador.test(modelo_dl, datamodule=carrega_dados)

    # Salvando o modelo
    torch.save(modelo_dl.state_dict(), 'modelo_dl.pth')

# Restaura stderr
sys.stderr = temp_stderr

# Limpa o arquivo temporário
temp_stderr.close()


