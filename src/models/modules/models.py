from torch import nn
import torchvision
from pytorch_lightning import LightningModule
import torchxrayvision as xrv

class Models(LightningModule):

    def __init__(self, hparams: dict):
        super().__init__()

        if hparams["model"].lower()=="rsna":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-rsna", 
                                            num_classes = hparams["num_classes"], 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="nih":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-nih", 
                                            num_classes = hparams["num_classes"], 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="padchest":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-pc", 
                                            num_classes = hparams["num_classes"], 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="chexpert":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-chex", 
                                            num_classes = hparams["num_classes"], 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="mimic_nb":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb", 
                                            num_classes = hparams["num_classes"], 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="mimic_ch":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch", 
                                            num_classes = hparams["num_classes"], 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="all":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all", 
                                            num_classes = hparams["num_classes"], 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower() == "densenet121":
            self.model = torchvision.models.densenet121(pretrained=hparams["pretrained"],
                                                        drop_rate=hparams["drop_rate"])

        elif hparams["model"].lower() == "densenet169":
            self.model = torchvision.models.densenet169(pretrained=hparams["pretrained"],
                                                        drop_rate=hparams["drop_rate"])

        elif hparams["model"].lower() == "densenet161":
            self.model = torchvision.models.densenet161(pretrained=hparams["pretrained"],
                                                        drop_rate=hparams["drop_rate"])

        elif hparams["model"].lower() == "densenet201":
            self.model = torchvision.models.densenet201(pretrained=hparams["pretrained"],
                                                        drop_rate=hparams["drop_rate"])

        #Modifying classifiers to current number of classes
        if hparams["model"].lower() in ["densenet121","densenet169","densenet161","densenet201", "googlenet"]:
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Linear(num_ftrs, 500),nn.Linear(500, hparams["num_classes"]))

    def forward(self, x):
        return self.model(x)
