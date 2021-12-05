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

        elif hparams["model"].lower()=="jfh":
            self.model = xrv.baseline_models.jfhealthcare.DenseNet(num_classes = hparams["num_classes"], 
                                                                    drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="all":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all", 
                                            num_classes = hparams["num_classes"], 
                                            drop_rate = hparams["drop_rate"])

        elif hparams["model"].lower()=="resnet18":
            self.model = torchvision.models.resnet18(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="resnet34":
            self.model = torchvision.models.resnet34(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="resnet50":
            self.model = torchvision.models.resnet50(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="resnet101":
            self.model = torchvision.models.resnet101(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="resnet152":
            self.model = torchvision.models.resnet152(pretrained=hparams["pretrained"])
        
        elif hparams["model"].lower()=="vgg11":
            self.model = torchvision.models.vgg11(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="vgg11bn":
            self.model = torchvision.models.vgg11_bn(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="vgg13":
            self.model = torchvision.models.vgg13(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="vgg13bn":
            self.model = torchvision.models.vgg13_bn(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="vgg16":
            self.model = torchvision.models.vgg16(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="vgg16bn":
            self.model = torchvision.models.vgg16_bn(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="vgg19":
            self.model = torchvision.models.vgg19(pretrained=hparams["pretrained"])

        elif hparams["model"].lower()=="vgg19bn":
            self.model = torchvision.models.vgg19_bn(pretrained=hparams["pretrained"])
        
        elif hparams["model"].lower()=="alexnet":
            self.model = torchvision.models.alexnet(pretrained=hparams["pretrained"])
            num_ftrs = self.model.classifier[4].out_features = 500
            self.model.classifier[6].in_features = 500
            self.model.classifier[6].out_features = hparams["num_classes"]
        
        elif hparams["model"].lower()=="squeezenet_1.0":
            self.model = torchvision.models.squeezenet1_0(pretrained=hparams["pretrained"])
            self.model.classifier[1] = nn.Conv2d(512, hparams["num_classes"], kernel_size=(1,1), stride=(1,1))


        elif hparams["model"].lower()=="squeezenet_1.1":
            self.model = torchvision.models.squeezenet1_1(pretrained=hparams["pretrained"])
            self.model.classifier[1] = nn.Conv2d(512, hparams["num_classes"], kernel_size=(1,1), stride=(1,1))
        
        elif hparams["model"].lower()=="googlenet":
            self.model = torchvision.models.googlenet(pretrained=hparams["pretrained"])
                                                    
        elif hparams["model"].lower()=="shufflenet":
            self.model = torchvision.models.shufflenet_v2_x1_0(pretrained=hparams["pretrained"])
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 500),
                                        nn.Linear(500, hparams["num_classes"]))

        elif hparams["model"].lower()=="mobilenet_v2":
            self.model = torchvision.models.mobilenet_v2(pretrained=hparams["pretrained"])
            self.model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                                nn.Linear(in_features=self.model.classifier[1].in_features, out_features=500),
                                                nn.Linear(in_features=500, out_features=hparams["num_classes"]))

        elif hparams["model"].lower() == "inception_v3":
            self.model = torchvision.models.inception_v3(pretrained=hparams["pretrained"])
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 500),
                                        nn.Linear(500, hparams["num_classes"]))

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

        else:
            print("ERROR: No model selected")


        #Modifying classifiers to current number of classes
        if hparams["model"].lower() in ["densenet121","densenet169","densenet161","densenet201", "googlenet"]:
            num_ftrs = self.model.classifier.in_features
            self.model.classifier.out_featres = hparams["num_classes"]
        
        elif hparams["model"].lower() in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]:
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=hparams["num_classes"])
        
        elif hparams["model"].lower() in ["VGG11", "VGG11BN", "VGG13", "VGG13BN", "VGG16", "VGG16BN", "VGG19", "VGG19BN"]:
            self.model.classifier[3]=nn.Linear(in_features=self.model.classifier[0].out_features,out_features=500)
            self.model.classifier[6]=nn.Linear(in_features=self.model.classifier[3].out_features,out_features=hparams["num_classes"])

    def forward(self, x):
        return self.model(x)
