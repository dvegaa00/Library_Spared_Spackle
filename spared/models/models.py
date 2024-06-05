import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(torch.nn.Module):
    def __init__(self, backbone, use_pretrained,  latent_dim):

        super(ImageEncoder, self).__init__()

        self.backbone = backbone
        self.use_pretrained = use_pretrained
        self.latent_dim = latent_dim

        # Initialize the model using various options 
        self.encoder, self.input_size = self.initialize_model()

    def initialize_model(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        model_weights = 'IMAGENET1K_V1' if self.use_pretrained else None
        input_size = 0

        if self.backbone == "resnet": ##
            """ Resnet18 acc@1 (on ImageNet-1K): 69.758
            """
            model_ft = models.resnet18(weights=model_weights)   #Get model
            num_ftrs = model_ft.fc.in_features                  #Get in features of the fc layer (final layer)
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)  #Keep in features, but modify out features for self.latent_dim
            input_size = 224                                    #Set input size of each image

        elif self.backbone == "resnet50":
            """ Resnet50 acc@1 (on ImageNet-1K): 76.13
            """
            model_ft = models.resnet50(weights=model_weights)   
            num_ftrs = model_ft.fc.in_features                  
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)  
            input_size = 224                                    

        elif self.backbone == "ConvNeXt":
            """ ConvNeXt tiny acc@1 (on ImageNet-1K): 82.52
            """
            model_ft = models.convnext_tiny(weights=model_weights)
            num_ftrs = model_ft.classifier[2].in_features
            model_ft.classifier[2] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "EfficientNetV2":
            """ EfficientNetV2 small acc@1 (on ImageNet-1K): 84.228
            """
            model_ft = models.efficientnet_v2_s(weights=model_weights)
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 384

        elif self.backbone == "InceptionV3":
            """ InceptionV3 acc@1 (on ImageNet-1K): 77.294
            """
            model_ft = models.inception_v3(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 299

        elif self.backbone == "MaxVit":
            """ MaxVit acc@1 (on ImageNet-1K): 83.7
            """
            model_ft = models.maxvit_t(weights=model_weights)
            num_ftrs = model_ft.classifier[5].in_features
            model_ft.classifier[5] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "MobileNetV3":
            """ MobileNet V3 acc@1 (on ImageNet-1K): 67.668
            """
            model_ft = models.mobilenet_v3_small(weights=model_weights)
            num_ftrs = model_ft.classifier[3].in_features
            model_ft.classifier[3] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "ResNetXt":
            """ ResNeXt-50 32x4d acc@1 (on ImageNet-1K): 77.618
            """
            model_ft = models.resnext50_32x4d(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224


        elif self.backbone == "ShuffleNetV2":
            """ ShuffleNetV2 acc@1 (on ImageNet-1K): 60.552
            """
            model_ft = models.shufflenet_v2_x0_5(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "ViT":
            """ Vision Transformer acc@1 (on ImageNet-1K): 81.072
            """
            model_ft = models.vit_b_16(weights=model_weights)
            num_ftrs = model_ft.heads.head.in_features
            model_ft.heads.head = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "WideResNet":
            """ Wide ResNet acc@1 (on ImageNet-1K): 78.468
            """
            model_ft = models.wide_resnet50_2(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "densenet": 
            """ Densenet acc@1 (on ImageNet-1K): 74.434
            """
            model_ft = models.densenet121(weights=model_weights)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224
        
        elif self.backbone == "swin": 
            """ Swin Transformer tiny acc@1 (on ImageNet-1K): 81.474
            """
            model_ft = models.swin_t(weights=model_weights)
            num_ftrs = model_ft.head.in_features
            model_ft.head = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def forward(self, tissue_tiles):

        latent_space = self.encoder(tissue_tiles)

        return latent_space