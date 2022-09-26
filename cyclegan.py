from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Tuple
from warnings import warn
from torchvision.utils import save_image
from datamodule import CustomDataset
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from callback import SRImageLoggerCallback
from pytorch_lightning.utilities.model_summary import ModelSummary
from datamodule import TVTDataModule
from datasets.utils import prepare_sr_datasets
from components import SRGANDiscriminator, SRGANGenerator, SRGAN_B, VGG19FeatureExtractor

from pathlib import Path


class CYCLEGAN(pl.LightningModule):

    def __init__(
            self,
            image_channels: int = 1,  # changed from 3
            feature_maps_gen: int = 64,
            feature_maps_disc: int = 64,
            num_res_blocks: int = 16,
            scale_factor: int = 4,
            generator_checkpoint: Optional[str] = None,
            learning_rate: float = 1e-4,
            scheduler_step: int = 100, reconstr_w=10, id_w=2,
            **kwargs: Any,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        self.reconstr_w = reconstr_w
        self.id_w = id_w

        # G is mapping from x to y and F is mapping from y to x.

        if generator_checkpoint:

            pass
            # self.generator = torch.load(generator_checkpoint)
            # m = SRResNet.load_from_checkpoint(checkpoint_path=generator_checkpoint, image_channels=1)
            # self.G = m.G
            # self.F = m.F
        else:
            assert scale_factor in [2, 4]
            num_ps_blocks = scale_factor // 2
            self.G = SRGANGenerator(image_channels, feature_maps_gen, num_res_blocks, num_ps_blocks)

            self.F = SRGAN_B(image_channels, feature_maps_gen, num_res_blocks, num_ps_blocks)

        self.Dx = SRGANDiscriminator(image_channels, feature_maps_disc)  # Lr
        self.Dy = SRGANDiscriminator(image_channels, feature_maps_disc)  # Hr

        self.mae = torch.nn.L1Loss()
        self.generator_loss = torch.nn.MSELoss()
        self.discriminator_loss = torch.nn.MSELoss()
        # self.vgg_feature_extractor = VGG19FeatureExtractor(image_channels)

    def configure_optimizers(self):

        G_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.hparams.learning_rate,
                                       betas=(0.5, 0.999))
        F_optimizer = torch.optim.Adam(self.F.parameters(), lr=self.hparams.learning_rate,
                                       betas=(0.5, 0.999))
        Dx_optimizer = torch.optim.Adam(self.Dx.parameters(), lr=self.hparams.learning_rate,
                                        betas=(0.5, 0.999))
        Dy_optimizer = torch.optim.Adam(self.Dy.parameters(), lr=self.hparams.learning_rate,
                                        betas=(0.5, 0.999))

        sched_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[self.hparams.scheduler_step], gamma=0.1)
        sched_F = torch.optim.lr_scheduler.MultiStepLR(F_optimizer, milestones=[self.hparams.scheduler_step], gamma=0.1)
        sched_Dx = torch.optim.lr_scheduler.MultiStepLR(Dx_optimizer, milestones=[self.hparams.scheduler_step],
                                                        gamma=0.1)
        Sched_Dy = torch.optim.lr_scheduler.MultiStepLR(Dy_optimizer, milestones=[self.hparams.scheduler_step],
                                                        gamma=0.1)

        return [G_optimizer, F_optimizer, Dx_optimizer,
                Dy_optimizer], [sched_G, sched_F, sched_Dx, Sched_Dy]

    def forward(self, lr_image: torch.Tensor) -> torch.Tensor:

        return self.G(lr_image)

    def training_step(self, batch, batch_idx, optimizer_idx):
        hr, lr = batch
        b = hr.size()[0]

        # valid = torch.ones(b, 1, 30, 30).cuda()
        # fake = torch.zeros(b, 1, 30, 30).cuda()

        # Train Generator
        if optimizer_idx == 0 or optimizer_idx == 1:

            result = self._gen_step(hr, lr)

        # Train Discriminator
        elif optimizer_idx == 2 or optimizer_idx == 3:

            result = self._disc_step(hr, lr)

        return result

    def _gen_step(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:

        # Perceptual loss
        fake_hr = self.G(lr_image)
        fake_pred_hr = self.Dy(fake_hr)
        # per_loss_A = self._perceptual_loss(hr_image, fake_hr)

        fake_lr = self.F(hr_image)
        fake_pred_lr = self.Dx(fake_lr)
        # per_loss_B = self._perceptual_loss(lr_image, fake_lr)

        # perceptual_loss = per_loss_A + per_loss_B

        # Gan loss
        adv_loss_A = self._adv_loss(fake_pred_hr, ones=True)
        adv_loss_B = self._adv_loss(fake_pred_lr, ones=True)
        adv_loss = (adv_loss_A + adv_loss_B) / 2

        # Content loss
        content_loss_A = self._content_loss(hr_image, fake_hr)
        content_loss_B = self._content_loss(lr_image, fake_lr)
        content_loss = content_loss_A + content_loss_B

        # Cyclic loss
        b = hr_image.size()[0]
        # valid = torch.ones(b, 1, 30, 30).cuda()

        # adv_loss_A = self._adv_loss(self.Dy(self.G(lr_image)), ones=True)
        # adv_loss_B = self._adv_loss(self.Dx(self.F(hr_image)), ones= True)
        # adv_loss = (adv_loss_A + adv_loss_B) /2

        # val_base = self.generator_loss(self.Dy(self.G(lr_image)), valid)
        # val_style = self.generator_loss(self.Dx(self.F(hr_image)), valid)
        # val_loss = (val_base + val_style) / 2

        # reconstr_base = self.mae(self.G(self.F(hr_image)), hr_image)
        # reconstr_style = self.mae(self.F(self.G(lr_image)), lr_image)
        # reconstr_loss = (reconstr_base + reconstr_style) / 2

        gen_loss = 0.001 * adv_loss + content_loss

        self.log("loss/gen", gen_loss, on_step=True, on_epoch=True)
        return gen_loss

    def _disc_step(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        b = hr_image.size()[0]
        # valid = torch.ones(b, 1, 30, 30).cuda()
        # fake = torch.zeros(b, 1, 30, 30).cuda()
        D_base_gen_loss = self._adv_loss(self.Dy(self.G(lr_image)), ones=False)
        D_style_gen_loss = self._adv_loss(self.Dx(self.F(hr_image)), ones=False)
        D_base_valid_loss = self._adv_loss(self.Dy(hr_image), ones=True)
        D_style_valid_loss = self._adv_loss(self.Dx(lr_image), ones=True)

        D_gen_loss = (D_base_gen_loss + D_style_gen_loss) / 2

        # Loss Weight
        disc_loss = (D_gen_loss + D_base_valid_loss + D_style_valid_loss) / 3

        # real_pred = self.discriminator(hr_image)
        # real_loss = self._adv_loss(real_pred, ones=True)

        # _, fake_pred = self._fake_pred(lr_image)
        # fake_loss = self._adv_loss(fake_pred, ones=False)

        # disc_loss = 0.5 * (real_loss + fake_loss)

        self.log("loss/disc", disc_loss, on_step=True, on_epoch=True)
        return disc_loss

    @staticmethod
    def _adv_loss(pred: torch.Tensor, ones: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if ones else torch.zeros_like(pred)
        adv_loss = F.binary_cross_entropy_with_logits(pred, target)
        return adv_loss

    # def _perceptual_loss(self, hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    #     real_features = self.vgg_feature_extractor(hr_image)
    #     fake_features = self.vgg_feature_extractor(fake)
    #     perceptual_loss = self._content_loss(real_features, fake_features)
    #     return perceptual_loss

    @staticmethod
    def _content_loss(hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(hr_image, fake)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--scheduler_step", default=100, type=float)
        return parser


def cli_main(args=None):
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="stl10", type=str, choices=["celeba", "mnist", "stl10"])
    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--scale_factor", default=4, type=int)
    parser.add_argument("--save_model_checkpoint", dest="save_model_checkpoint", action="store_true")

    parser = TVTDataModule.add_argparse_args(parser)
    parser = CYCLEGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args(args)

    # datasets = prepare_sr_datasets(args.dataset, args.scale_factor, args.data_dir)
    # dm = TVTDataModule(*datasets, **vars(args))
    dm = CustomDataset(args.scale_factor, image_tpe='.tif', patch_size=128, batch_size=16)

    # SPECIFY CHECKPOINT HERE

    # generator_checkpoint = Path('lightning_logs/srresnet/stl10-scale_factor=4/checkpoints/epoch=49-step=3550.ckpt')
    # if not generator_checkpoint.exists():
    #     warn(
    #         "No generator checkpoint found. Training generator from scratch. \
    #         Use srresnet_module.py to pretrain the generator."
    #     )
    #     generator_checkpoint = None

    # LOADING MODEL

    model = CYCLEGAN(
        **vars(args), image_channels=dm.image_channels, generator_checkpoint=None
    )  # change from dm.dataset_test.image_channels

    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="gpu",
        devices=1,
        max_epochs=200,
        callbacks=[SRImageLoggerCallback(log_interval=args.log_interval, scale_factor=args.scale_factor)],
        logger=pl.loggers.TensorBoardLogger(
            save_dir="lightning_logs",
            name="cyclegan",
            version='train',
            # version=f"{args.dataset}-scale_factor={args.scale_factor}",
            default_hp_metric=False,
        ),
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    cli_main()